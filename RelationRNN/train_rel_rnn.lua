require '..'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a Recurrent Neural Network to embed a sentence')
cmd:text()
cmd:text('Options')

cmd:option('-vocabSize',100002,'number of words in dictionary')

cmd:option('-relSize',7524,'number of relations in dictionary')
cmd:option('-relEmbedSize',256,'size of rel embedding')

cmd:option('-wrdEmbedSize',300,'size of word embedding')
cmd:option('-wrdEmbedPath','../embedding/word.100k.glove.t7','pretained word embedding path')

cmd:option('-numLayer',2,'number of RNN layers')
cmd:option('-maxSeqLen',40,'number of timesteps to unroll to')
cmd:option('-hiddenSize',256,'size of RNN internal state')
cmd:option('-dropoutRate',0.5,'dropout rate')

cmd:option('-negSize',1024,'number of negtive samples for each iteration')
cmd:option('-maxEpochs',1000,'number of full passes through the training data')
cmd:option('-initRange',0.08,'the range of uniformly initialize parameters')
cmd:option('-costMargin',0.1,'the margin used in the ranking cost')
cmd:option('-useGPU',1,'whether to use gpu for computation')

cmd:option('-printEvery',100,'how many steps/minibatches between printing out the loss')
cmd:option('-saveEvery',100,'how many epochs between auto save trained models')
cmd:option('-saveFile','model.rel.stackBiRNN','filename to autosave the model (protos) to')
cmd:option('-logFile','logs/rel.stackBiRNN.log','log file to record training information')
cmd:option('-dataFile', '../data/train.relation_ranking.t7','training data file')

cmd:option('-seed',123,'torch manual random number generator seed')
cmd:text()

----------------------------- parse params -----------------------------

local opt = cmd:parse(arg)
-- local flog = logroll.file_logger(opt.logFile)
local flog = logroll.print_logger()
if opt.useGPU > 0 then
    cutorch.setDevice(opt.useGPU)
    torch.setdefaulttensortype('torch.CudaTensor')
end

----------------------------- define loader -----------------------------
local loader = SeqRankingLoader(opt.dataFile, opt.negSize, opt.relSize, flog)

----------------------------- define models -----------------------------
-- word embedding model
local wordEmbed = cudacheck(nn.LookupTable(opt.vocabSize, opt.wrdEmbedSize))
-- loadPretrainedEmbed(wordEmbed, opt.wrdEmbedPath)

-- rel embedding model
-- local relEmbed = torch.load('../TransE/model.60').RelEmbed
local relEmbed = cudacheck(nn.LookupTable(opt.relSize, opt.relEmbedSize))
relEmbed.weight:uniform(-opt.initRange, opt.initRange)
relEmbed.weight:renorm(2, 2, 1)

local posRelDrop = nn.Dropout(0.3)
local negRelDrop = nn.Dropout(0.3)

-- multi-layer (stacked) Bi-RNN
local config = {}
config.hiddenSize = opt.hiddenSize
config.maxSeqLen  = opt.maxSeqLen
config.maxBatch   = 256
config.logger     = flog

local RNN = {}
for l = 1, opt.numLayer do
    config.inputSize = l == 1 and opt.wrdEmbedSize or opt.hiddenSize * 2
    RNN[l] = BiGRU(config)
end

local selectLayer = BiRNNSelect()
local linearLayer = nn.Linear(2 * opt.hiddenSize, opt.relEmbedSize)

local seqModel = nn.Sequential()
seqModel:add(wordEmbed)
for l = 1, opt.numLayer do
    seqModel:add(nn.Dropout(opt.dropoutRate))
    seqModel:add(RNN[l])
end
seqModel:add(selectLayer)
seqModel:add(linearLayer)

-- ranking score model
local scoreModel = TripleScore(opt.negSize)

-- put all models together
local model = {}
model.seqModel   = seqModel
model.relEmbed   = relEmbed
model.posRelDrop = posRelDrop
model.negRelDrop = negRelDrop
model.scoreModel = scoreModel

-- margin ranking criterion
local criterion  = nn.MarginRankingCriterion(opt.costMargin)

-- put together parms and grad pointers in optimParams and optimGrad tables
local optimParams, optimGrad = {}, {}
for l = 1, opt.numLayer do
    local rnnParams, rnnGrad = RNN[l]:getParameters()
    rnnParams:uniform(-opt.initRange, opt.initRange)
    optimParams[l], optimGrad[l] = rnnParams, rnnGrad
end
optimParams[#optimParams+1], optimGrad[#optimGrad+1] = linearLayer:getParameters()

-- optimization configurations [subject to change]
local lrWrd, lrRel = 1e-3, 3e-4

local optimConf = {['lr'] = {}, ['momentum'] = 0.3}
-- local optimConf = {['lr'] = {}}
for l = 1, #optimParams do optimConf['lr'][l] = 1e-3 end
local optimizer = AdaGrad(optimGrad, optimConf)

-- prepare for training
local sumLoss, epochLoss  = 0, 0
local maxIters = opt.maxEpochs * loader.numBatch
local ones = torch.ones(loader.batchSize, loader.negSize)

-- core training loop
for i = 1, maxIters do
    xlua.progress(i, maxIters)
    -- in the beginning of each loop, clean the grad_params
    relEmbed:zeroGradParameters()
    wordEmbed:zeroGradParameters()
    for i = 1, #optimGrad do optimGrad[i]:zero() end

    ----------------------- load minibatch ------------------------
    local seq, pos, negs = loader:nextBatch()
    local currSeqLen = seq:size(1)
    local loss = 0

    ------------------------ forward pass -------------------------
    -- sequence vectors [n_batch x n_dim]
    local seqVec = seqModel:forward(seq)

    -- positive vectors [n_batch x n_dim]
    local posVec = relEmbed:forward(pos):clone()
    local posDropVec = posRelDrop:forward(posVec)

    -- negative matrix  [n_neg x n_batch x n_dim]
    local negMat = relEmbed:forward(negs)
    local negDropMat = negRelDrop:forward(negMat)

    -- scores table {[1] = postive_scores, [2] = negative_scores}
    -- local scores = scoreModel:forward({seqVec, posVec, negMat})
    local scores = scoreModel:forward({seqVec, posDropVec, negDropMat})
    local loss = criterion:forward(scores, ones)
    
    -- d_scores table {[1] = d_postive_scores, [2] = d_negative_scores}
    local d_scores = criterion:backward(scores, ones)

    -- d_seqVec [n_batch x n_dim], d_posVec [n_batch x n_dim], d_negMat [n_neg x n_batch x n_dim]
    -- local d_seqVec, d_posVec, d_negMat = unpack(scoreModel:backward({seqVec, posVec, negMat}, d_scores))
    local d_seqVec, d_posDropVec, d_negDropMat = unpack(scoreModel:backward({seqVec, posDropVec, negDropMat}, d_scores))

    local d_negMat = negRelDrop:backward(negMat, d_negDropMat)

    local d_posVec = posRelDrop:backward(posVec, d_posDropVec)

    -- grad due to negative matrix
    relEmbed:backward(negs, d_negMat)

    -- grad due to positive vectors
    relEmbed:backward(pos, d_posVec)

    -- grad to the sequence model
    -- seqModel:backward(dropedSeq, d_seqVec)
    seqModel:backward(seq, d_seqVec)

    ----------------------- parameter update ----------------------
    -- sgd with scheduled anealing
    relEmbed:updateParameters(lrRel / (1 + 0.0001 * i))

    -- renorm rel embeding into normal ball
    relEmbed.weight:renorm(2, 2, 1)

    -- sgd with scheduled anealing (override with sparse update)
    wordEmbed:updateParameters(lrWrd / (1 + 0.0001 * i))

    -- adagrad for rnn, projection    
    for l = 1, opt.numLayer do optimGrad[l]:clamp(-10, 10) end
    optimizer:updateParams(optimParams, optimGrad)
    
    -- accumulate loss
    sumLoss   = sumLoss + loss
    epochLoss = epochLoss + loss

    -- scheduled anealing the momentum rate after each epoch
    if i % loader.numBatch == 0 then
        flog.info(string.format('epoch %3d, loss %6.8f', i / loader.numBatch, epochLoss / loader.numBatch / loader.negSize))
        epochLoss = 0
        if i / loader.numBatch >= 10 then
            optimizer:updateMomentum(math.min(optimizer.momentum + 0.3, 0.99))
        end        
    end

    ------------------------ training info ------------------------
    if i % opt.printEvery == 0 then
        flog.info(string.format("iter %4d, loss = %6.8f", i, sumLoss / opt.printEvery / opt.negSize))
        sumLoss = 0
    end
    if i % (loader.numBatch * opt.saveEvery) == 0 then
        -- save model after each epoch
        local epoch = i / loader.numBatch
        print('saving model after epoch', epoch)
        torch.save(opt.saveFile..'.'..opt.useGPU..'.'..epoch, model)
    end
end
