require '..'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a Recurrent Neural Network to classify a sequence of words')
cmd:text()
cmd:text('Comandline Options')

cmd:option('-wordVocab','../vocab/vocab.word.t7','training data file')
cmd:option('-testData','inference-data/label.valid.t7','data file to predict')
cmd:option('-modelFile','model/model.BiGRU','path to the trained model')

cmd:option('-useGPU',1,'which GPU is used for computation')

cmd:text()

----------------------------- Basic Options -----------------------------

local opt = cmd:parse(arg)

local wordVocab = torch.load(opt.wordVocab)

if opt.useGPU > 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.useGPU)
    torch.setdefaulttensortype('torch.CudaTensor')
end

----------------------------- Data Loader -----------------------------
local loader = SeqLabelingLoader(opt.testData, flog)

-------------------------- Load & Init Models -------------------------
local model = torch.load(opt.modelFile)
local seqModel = model.seqModel
local linearCRF = model.linearCRF
seqModel:evaluate()
linearCRF:evaluate()

----------------------------- Prediction -----------------------------
local maxIters = loader.numBatch

local fields = stringx.split(opt.testData, '.')
local split = fields[#fields-1]
local file = io.open(string.format("label.result.%s", split), 'w')

for i = 1, maxIters do
    xlua.progress(i, maxIters)

    ----------------------- load minibatch ------------------------
    local seq, _ = loader:nextBatch(1)
    local currSeqLen = seq:size(1)
    local seqVec = seqModel:forward(seq)
    local predict = linearCRF:forward(seqVec)

    for i = 1, currSeqLen do
        file:write(predict[{i,1}]-0.999, ' ')
    end
    file:write('\n')
end
file:close()