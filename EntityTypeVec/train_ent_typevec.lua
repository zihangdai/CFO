require '..'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a Recurrent Neural Network to classify a sequence of words')
cmd:text()
cmd:text('Comandline Options')

cmd:option('-wordVocabSize',100003,'number of words in dictionary')
cmd:option('-wordEmbedDim',300,'size of word embedding')
cmd:option('-wordEmbedPath','../embedding/word.100k.glove.t7','pretained word embedding path')

cmd:option('-hiddenSize',256,'size of BiGRU unit')
cmd:option('-outputType',1,'output type of each rnn layer')
cmd:option('-numLayer',2,'number of BiGRU layers')
cmd:option('-maxSeqLen',200,'number of steps the BiGRU needs to unroll')

cmd:option('-numClass',1,'number of classes in classification')

cmd:option('-trainData','../data/train.entity_typevec.t7','training data file')

cmd:option('-optMethod','adamomentum','the optimization method used')
cmd:option('-initRange',0.08,'the range of uniformly initialize parameters')
cmd:option('-momentumEpoch',1,'after which epoch, the model starts to increase momentum')
cmd:option('-maxEpochs',500,'number of full passes through the training data')

cmd:option('-printEvery',100,'the frequency (# minibatches) of logging loss information')
cmd:option('-logFile','logs/log.BiGRU','log file to record training information')
cmd:option('-saveEvery',100,'the frequency (# epochs) of automatic saving trained models')
cmd:option('-saveFile','model.BiGRU','filename for saving trained model')

cmd:option('-useGPU',1,'which GPU is used for computation')

cmd:text()

----------------------------- Basic Options -----------------------------

local opt = cmd:parse(arg)
-- local flog = logroll.file_logger(opt.logFile)
local flog = logroll.print_logger()

if opt.useGPU > 0 then
    cutorch.setDevice(opt.useGPU)
    torch.setdefaulttensortype('torch.CudaTensor')
end

----------------------------- Data Loader -----------------------------
local loader  = SeqMultiLabelLoader(opt.trainData, flog)

----------------------------- Init Models -----------------------------
-- Init word embedding model
local wordEmbed = cudacheck(nn.LookupTable(opt.wordVocabSize, opt.wordEmbedDim))
-- loadPretrainedEmbed(wordEmbed, opt.wordEmbedPath)

-- Init Stacked BiGRU
local rnnconfig = {
	hiddenSize = opt.hiddenSize,
	maxSeqLen = opt.maxSeqLen, 
	maxBatch = loader.batchSize, 
	logger = flog
}

local RNN = {}
for l = 1, opt.numLayer do
    rnnconfig.inputSize = l == 1 and opt.wordEmbedDim or opt.hiddenSize * 2
    RNN[l] = BiGRU(rnnconfig)
end

-- Init the Classification Criterion
local criterion  = nn.BCECriterion()

local selectLayer = BiRNNSelect()
local linearLayer = nn.Linear(2 * opt.hiddenSize, 501)

local model = nn.Sequential()
model:add(wordEmbed)
for l = 1, opt.numLayer do
    model:add(nn.Dropout(0.3))
	model:add(RNN[l])
end
model:add(selectLayer)
model:add(linearLayer)
model:add(nn.Sigmoid())

----------------------------- Optimization -----------------------------
-- Create tables to hold params and grads
local optimParams, optimGrads = {}, {}
for l = 1, opt.numLayer do
    optimParams[l], optimGrads[l] = RNN[l]:getParameters()
    optimParams[l]:uniform(-opt.initRange, opt.initRange)
end
optimParams[#optimParams+1], optimGrads[#optimGrads+1] = wordEmbed:getParameters()
optimParams[#optimParams+1], optimGrads[#optimGrads+1] = linearLayer:getParameters()


-- Configurations for Optimizer
local optimizer
if opt.optMethod == 'adamomentum' then
    local optimConf = {lr = {}, momentum = 0.9, logger = flog}
    for l = 1, #optimParams do optimConf['lr'][l] = 1e-2 end
    optimizer = AdaGrad(optimGrads, optimConf)
elseif opt.optMethod == 'adagrad' then
    local optimConf = {lr = {}, logger = flog}
    for l = 1, #optimParams do optimConf['lr'][l] = 2e-2 end
    optimizer = AdaGrad(optimGrads, optimConf)
elseif opt.optMethod == 'momentum' then
    local optimConf = {lr = {}, momentum = 0.9, annealing = 0.01, logger = flog}
    for l = 1, #optimParams do optimConf['lr'][l] = 3e-1 end
    optimizer = SGD(optimGrads, optimConf)
elseif opt.optMethod == 'SGD' then
    local optimConf = {lr = {}, annealing = 0.01, logger = flog}
    for l = 1, #optimParams do optimConf['lr'][l] = 5e-3 end
    optimizer = SGD(optimGrads, optimConf)
else 
    print ('Error: optMethod not match')
    os.exit(-1)
end

local lrWrd = 1e-4

----------------------------- Training -----------------------------
local sumLoss  = 0
local sumCorr  = 0
local sumTrue  = 0
local maxIters = opt.maxEpochs * loader.numBatch
flog.info(string.rep('-', 40))
flog.info('Begin Training')

for i = 1, maxIters do
    xlua.progress(i, maxIters)

    ----------------------- clean gradients -----------------------
    for i = 1, #optimGrads do optimGrads[i]:zero() end

    ----------------------- load minibatch ------------------------
    local seq, labels = loader:nextBatch()
    local currSeqLen = seq:size(1)

    ------------------------ forward pass -------------------------
    local predict = model:forward(seq)

    -------------------------- criterion --------------------------
    local loss = criterion:forward(predict, labels)    
    sumLoss = sumLoss + loss

    local hardPred = torch.ge(predict, 0.5)
    sumCorr = sumCorr + torch.cmul(hardPred:type(torch.type(labels)), labels):sum()
    sumTrue = sumTrue + labels:sum()

    ------------------------ backward pass ------------------------
    local d_predict = criterion:backward(predict, labels)
    model:backward(seq, d_predict)    
    
    ----------------------- parameter update ----------------------
    -- optim for rnn, projection
    for l = 1, opt.numLayer do optimGrads[l]:clamp(-10, 10) end
    optimizer:updateParams(optimParams, optimGrads)

    -- Logging 
    if i % loader.numBatch == 0 then
        flog.info(string.format("finish epoch %d", i / loader.numBatch))
    end

    ------------------------ training info ------------------------
    if i % opt.printEvery == 0 then
        flog.info(string.format("iter %4d, loss = %5f, corr = %5f", 
            i, sumLoss / opt.printEvery, sumCorr / sumTrue))
        sumLoss, sumCorr, sumTrue = 0, 0, 0
    end
    if i % (loader.numBatch * opt.saveEvery) == 0 then
        local epoch = i / loader.numBatch
        print('Saving model after epoch ' .. epoch)
        torch.save(opt.saveFile..'.'..opt.useGPU..'.'..epoch, model)
    end
end
