require '..'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a Recurrent Neural Network to classify a sequence of words')
cmd:text()
cmd:text('Comandline Options')

cmd:option('-testData','data/valid.torch','training data file')
cmd:option('-modelFile','model.BiGRU','filename for loading trained model')

cmd:option('-useGPU',1,'which GPU is used for computation')

cmd:text()

----------------------------- Basic Options -----------------------------

local opt = cmd:parse(arg)
local flog = logroll.print_logger()

if opt.useGPU > 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.useGPU)
    torch.setdefaulttensortype('torch.CudaTensor')
    flog.info(string.rep('-', 50))
    flog.info('Set default tensor type to CudaTensor')
end

----------------------------- Data Loader -----------------------------
local loader = SeqMultiLabelLoader(opt.testData, flog)

-------------------------- Load & Init Models -------------------------
cutorch.reserveStreams(2)
local model = torch.load(opt.modelFile)
model:evaluate()

----------------------------- Prediction -----------------------------
local maxIters = loader.numBatch
flog.info(string.rep('-', 40))
flog.info('Begin Prediction')

local sumPred, sumCorr, sumTrue = 0, 0, 0

for i = 1, maxIters do
    xlua.progress(i, maxIters)

    ----------------------- load minibatch ------------------------
    local seq, labels = loader:nextBatch()
    local currSeqLen = seq:size(1)    

    local predict = model:forward(seq)
    local hardPred = torch.ge(predict, 0.5)
    sumCorr = sumCorr + torch.cmul(hardPred:type(torch.type(labels)), labels):sum()
    sumTrue = sumTrue + labels:sum()
    sumPred = sumPred + hardPred:sum()
    
end

local p, r = sumCorr / sumPred, sumCorr / sumTrue
print(p, r, 2 * p * r / (p + r))
