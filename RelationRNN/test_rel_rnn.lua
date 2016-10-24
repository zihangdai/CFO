require '..'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Testing a Recurrent Neural Network to embed a sentence')
cmd:text()
cmd:text('Options')
cmd:option('-useGPU',1,'whether to use gpu for computation')
cmd:option('-modelFile','model.rel.stackBiRNN','file path for saved model')
cmd:option('-testData','valid.single.label','run test on which data set')
cmd:option('-KG','FB5M','suffix appended to the output file name')
cmd:text()

-- parse input params
local opt = cmd:parse(arg)
local flog = logroll.print_logger()

if opt.useGPU > 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.useGPU)
    torch.setdefaulttensortype('torch.CudaTensor')    
end

-- load all models
local suffix = stringx.split(opt.modelFile, '.')
suffix = suffix[#suffix]
local model = torch.load(opt.modelFile)

-- init data loader and output files
local loader = RankingDataLoader(string.format('data/%s.%s.torch', opt.testData, opt.KG), flog)
local score_file = io.open(string.format('score.%s.%s', opt.testData, opt.KG, suffix), 'w')
local rank_file  = io.open(string.format('rank.%s.%s', opt.testData, opt.KG, suffix), 'w')

-- extract sub models
local relEmbed   = model.relEmbed
local seqModel   = model.seqModel
local scoreModel = model.scoreModel
local negRelDrop = model.negRelDrop

seqModel:evaluate()
negRelDrop:evaluate()

-- core testing loop
for i = 1, loader.numBatch do
    xlua.progress(i, loader.numBatch)
    ----------------------- load minibatch ------------------------
    local seq, pos, neg = loader:nextBatch(1)
    neg = neg:view(-1)
    local currSeqLen = seq:size(1)
    local loss = 0

    ------------------------ forward pass -------------------------    
    -- sequence vectors [n_batch x n_dim]
    local seqVec = seqModel:forward(seq)

    -- negative matrix  [n_neg x n_batch x n_dim]
    -- local negMat = relEmbed:forward(neg)
    
    local tmp = relEmbed:forward(neg)
    local negMat = negRelDrop:forward(tmp)
    
    -- sequence matrix  [n_neg x n_batch x n_dim]
    local seqMat = torch.repeatTensor(seqVec, negMat:size(1), 1)

    if opt.useGPU > 0 then
        scores = torch.cmul(seqMat, negMat):sum(2):view(-1)
    else
        scores = torch.mm(seqMat, negMat:t()):diag()
    end
    
    -- write to rank file
    if scores:size(1) > 1 then
        local _, argSort = scores:sort(1, true)

        rank_file:write(pos[1], '\t')
        for i = 1, argSort:size(1) do
            rank_file:write(neg[argSort[i]], ' ')
        end
        rank_file:write('\n')

        -- write to score file
        local topIndices = {}
        for i = 1, argSort:size(1) do
            topIndices[argSort[i]] = 1
        end
        for i = 1, scores:size(1) do
            if topIndices[i] then
                score_file:write(scores[i], ' ')
            else
                score_file:write(0, ' ')
            end
        end
        score_file:write('\n')
    else
        rank_file:write(pos[1], '\t')
        rank_file:write(neg[1])
        rank_file:write('\n')
        score_file:write(scores[1])
        score_file:write('\n')
    end

    collectgarbage()
end
score_file:close()
rank_file:close()
