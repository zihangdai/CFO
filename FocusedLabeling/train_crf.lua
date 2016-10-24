require '..'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a Recurrent Neural Network to classify a sequence of words')
cmd:text()
cmd:text('Comandline Options')

cmd:option('-wordVocabSize',100002,'number of words in dictionary')
cmd:option('-wordEmbedDim',300,'size of word embedding')
cmd:option('-wordEmbedPath','../embedding/word.100k.glove.t7','pretained word embedding path')

cmd:option('-hiddenSize',256,'size of BiGRU unit')
cmd:option('-outputType',1,'output type of each rnn layer')
cmd:option('-numLayer',2,'number of BiGRU layers')
cmd:option('-maxSeqLen',40,'number of steps the BiGRU needs to unroll')

cmd:option('-numClass',2,'number of classes in classification')

cmd:option('-trainData','../data/train.focused_labeling.t7','training data file')

cmd:option('-initRange',0.08,'the range of uniformly initialize parameters')
cmd:option('-momentumEpoch',1,'after which epoch, the model starts to increase momentum')
cmd:option('-maxEpochs',100,'number of full passes through the training data')

cmd:option('-printEvery',50,'the frequency (# minibatches) of logging loss information')
cmd:option('-logFile','logs/log.BiGRU','log file to record training information')
cmd:option('-saveEvery',10,'the frequency (# epochs) of automatic saving trained models')
cmd:option('-saveFile','model.BiGRU','filename for saving trained model')

cmd:option('-useGPU',1,'which GPU is used for computation')

cmd:text()

----------------------------- Basic Options -----------------------------

local opt = cmd:parse(arg)
local flog = logroll.file_logger(opt.logFile)
-- local flog = logroll.print_logger()

if opt.useGPU > 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.useGPU)
    torch.setdefaulttensortype('torch.CudaTensor')
    flog.info(string.rep('-', 50))
    flog.info('Set default tensor type to CudaTensor')
    torch.manualSeed(1)
    cutorch.manualSeed(1)
end

----------------------------- Data Loader -----------------------------
local loader = SeqLabelingLoader(opt.trainData, flog)

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

-- Init linear project model
local linear = Linear(opt.hiddenSize*2, opt.numClass)

-- Init the linear CRF
local linearCRF = CRF(opt.numClass, opt.maxSeqLen, loader.batchSize)

local seqModel = nn.Sequential()
seqModel:add(wordEmbed)
for l = 1, opt.numLayer do
	seqModel:add(nn.Dropout(0.7))
	seqModel:add(RNN[l])
end
seqModel:add(linear)

local model = {}
model.seqModel = seqModel
model.linearCRF = linearCRF

----------------------------- Optimization -----------------------------
-- Create tables to hold params and grads
local optimParams, optimGrads = {}, {}
for l = 1, opt.numLayer do
    optimParams[l], optimGrads[l] = RNN[l]:getParameters()    
end
optimParams[#optimParams+1], optimGrads[#optimGrads+1] = linear:getParameters()
optimParams[#optimParams+1], optimGrads[#optimGrads+1] = linearCRF:getParameters()
for i = 1, #optimParams do
    optimParams[i]:uniform(-opt.initRange, opt.initRange)
end
optimParams[#optimParams+1], optimGrads[#optimGrads+1] = wordEmbed:getParameters()
print(optimParams, optimGrads)

-- Configurations for Optimizer
local optimConf = {lr = {}, logger = flog}
for l = 1, #optimParams do optimConf['lr'][l] = 2e-2 end    
local optimizer = AdaGrad(optimGrads, optimConf)

----------------------------- Training -----------------------------

local avgProb = 0

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
    local seqVec = seqModel:forward(seq)
    local prob = linearCRF:forward({seqVec, labels})
    avgProb = avgProb + torch.mean(prob)
	
    ------------------------ backward pass ------------------------
    local d_seqVec = linearCRF:backward({seqVec, labels})    
    seqModel:backward(seq, d_seqVec)
    
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
        linearCRF:evaluate()
        local pred = linearCRF:forward(seqVec)
        local maskPred = torch.eq(pred, 2)
        local maskTrue = torch.eq(labels, 2)
        local corr = torch.eq(pred:type(torch.type(labels)), labels):cmul(maskTrue):sum()

        local p, r = corr / maskPred:sum(), corr / maskTrue:sum()
        flog.info(string.format("iter %4d, avg prob = %5f, p = %3f, r = %3f, F1 = %3f", i, avgProb / opt.printEvery, p, r, 2 * p * r / (p + r)))
        linearCRF:training()
        avgProb = 0
    end


    if i % (loader.numBatch * opt.saveEvery) == 0 then
        local epoch = i / loader.numBatch
        print('Saving model after epoch ' .. epoch)
        torch.save(opt.saveFile..'.'..opt.useGPU..'.'..epoch, model)
    end
end
