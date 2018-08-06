require '..'
require 'SeqLabelRankLoader'

local cmd = torch.CmdLine()
cmd:text('Comandline Options')
cmd:option('-testData','inference-data/ent.valid.t7','training data file')
cmd:option('-modelFile','model.BiGRU','filename for loading trained model')
cmd:option('-useGPU',0,'which GPU is used for computation')

cmd:text()

----------------------------- Basic Options -----------------------------

local opt = cmd:parse(arg)
local flog = logroll.print_logger()

if opt.useGPU > 0 then
    cutorch.setDevice(opt.useGPU)
    torch.setdefaulttensortype('torch.CudaTensor')
    flog.info(string.rep('-', 50))
    flog.info('Set default tensor type to CudaTensor')
end

----------------------------- Data Loader -----------------------------
local fields = stringx.split(opt.testData, '.')
local split = fields[#fields-1]
local loader = SeqLabelRankLoader(opt.testData, flog)
local score_file = io.open(string.format('score.ent.multi.%s', split), 'w')
local rank_file  = io.open(string.format('rank.ent.multi.%s', split), 'w')

-------------------------- Load & Init Models -------------------------
cutorch.reserveStreams(2)
local model = torch.load(opt.modelFile)
model:evaluate()

----------------------------- Prediction -----------------------------
local maxIters = loader.numBatch
flog.info(string.rep('-', 40))
flog.info('Begin Prediction')

for i = 1, maxIters do
    xlua.progress(i, maxIters)

    ----------------------- load minibatch ------------------------
    local seq, posIdx, candi = loader:nextBatch(1)
    local currSeqLen = seq:size(1)    
    local numCandi = candi:size(1)

    local predict = model:forward(seq)
    predict:maskedSelect(torch.lt(predict, 0.5)):zero()
    local repPred = predict:expandAs(candi)

    candi = candi:cuda()
    local scores = torch.cmul(repPred, candi):sum(2):view(numCandi)

    local _, argSort = torch.sort(scores, 1, true)
    rank_file:write(posIdx, '\t')
    for i = 1, numCandi do
        rank_file:write(argSort[i], ' ')
    end
    rank_file:write('\n')

    for i = 1, numCandi do
        score_file:write(scores[i], ' ')
    end
    score_file:write('\n')
end
rank_file:close()
score_file:close()
