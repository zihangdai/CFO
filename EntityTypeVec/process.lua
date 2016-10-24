require '..'

function multiLabelData()
    wordVocab = torch.load('../vocab/vocab.word.t7')

    trainDir = '../SimpleQuestions/EntityTypeVecData'
    createSeqMultiLabelData(trainDir..'/train.seq-types.txt', 'data/train.torch', wordVocab, 501, 256)

    createSeqLabelRankData('../Inference/FB5M-ngram/type.multi.valid.txt', 'data/valid.ngram.FB5M.torch', wordVocab, 501)
    createSeqLabelRankData('../Inference/FB5M-label/type.multi.valid.txt', 'data/valid.label.FB5M.torch', wordVocab, 501)
    createSeqLabelRankData('../Inference/FB5M-heuristics/type.multi.valid.txt', 'data/valid.heuristics.FB5M.torch', wordVocab, 501)

    createSeqLabelRankData('../Inference/5M-ngram/type.multi.test.txt', 'data/test.ngram.FB5M.torch', wordVocab, 501)
    createSeqLabelRankData('../Inference/5M-label/type.multi.test.txt', 'data/test.label.FB5M.torch', wordVocab, 501)
    createSeqLabelRankData('../Inference/5M-heuristics/type.multi.test.txt', 'data/test.heuristics.FB5M.torch', wordVocab, 501)
end

multiLabelData()