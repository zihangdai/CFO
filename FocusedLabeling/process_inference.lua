require '..'

local cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-testSplit','valid','use which data split set')
cmd:text()

local opt = cmd:parse(arg)

local wordVocab = torch.load('../vocab/vocab.word.t7')

local txtPath = string.format('inference-data/label.%s.txt', opt.testSplit)
local thPath = string.format('inference-data/label.%s.t7', opt.testSplit)

createSeqLabelingData(txtPath, thPath, wordVocab, 1)