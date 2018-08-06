require '..'

local cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-testSplit','valid','use which data split set')
cmd:text()

wordVocab = torch.load('../vocab/vocab.word.t7')
relationVocab = torch.load('../vocab/vocab.rel.t7')

txtSPath = string.format('../Inference/valid/rel.single.%s.txt', opt.testSplit)
txtMPath = string.format('../Inference/valid/rel.multi.%s.txt', opt.testSplit)

thSPath = string.format('inference-data/rel.single.%s.t7', opt.testSplit)
thMPath = string.format('inference-data/rel.multi.%s.t7', opt.testSplit)

createRankingData(txtSPath, thSPath, wordVocab, relationVocab, 1)
createRankingData(txtMPath, thMPath, wordVocab, relationVocab, 1)
