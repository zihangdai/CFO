require '..'

local cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-testSplit','valid','use which data split set')
cmd:text()

wordVocab = torch.load('../vocab/vocab.word.t7')
relationVocab = torch.load('../vocab/vocab.rel.t7')

txtPath = string.format('../Inference/FB5M-ngram/type.multi.%s.txt', opt.testSplit)
thPath = string.format('inference-data/ent.%s.t7', opt.testSplit)

createSeqLabelRankData(txtPath, thPath, wordVocab, 501)