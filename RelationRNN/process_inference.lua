require '..'

local cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-testSplit','valid','use which data split set')
cmd:text()

local opt = cmd:parse(arg)

local wordVocab = torch.load('../vocab/vocab.word.t7')
local relationVocab = torch.load('../vocab/vocab.rel.t7')

local txtSPath = string.format('../Inference/valid/rel.single.%s.txt', opt.testSplit)
local txtMPath = string.format('../Inference/valid/rel.multi.%s.txt', opt.testSplit)

local thSPath = string.format('inference-data/rel.single.%s.t7', opt.testSplit)
local thMPath = string.format('inference-data/rel.multi.%s.t7', opt.testSplit)

createRankingData(txtSPath, thSPath, wordVocab, relationVocab, 1)
createRankingData(txtMPath, thMPath, wordVocab, relationVocab, 1)
