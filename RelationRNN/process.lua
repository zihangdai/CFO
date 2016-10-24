require '..'

function trainData()
    wordVocab = torch.load('../vocab/vocab.word.t7')
    relationVocab = torch.load('../vocab/vocab.rel.t7')
    trainDir = '../SimpleQuestions/TextBasedRelationRanking'

    createSeqRankingData(trainDir..'/data.train.256', 'data/train.torch', wordVocab, relationVocab, 256)
end

function testData()
    wordVocab = torch.load('../vocab/vocab.word.t7')
    relationVocab = torch.load('../vocab/vocab.rel.t7')

    createRankingData('../Inference/FB5M-label/rel.single.valid.txt', 'data/valid.single.label.FB5M.torch', wordVocab, relationVocab, 1)
    createRankingData('../Inference/FB5M-label/rel.multi.valid.txt',  'data/valid.multi.label.FB5M.torch',  wordVocab, relationVocab, 1)

    createRankingData('../Inference/FB5M-ngram/rel.single.valid.txt', 'data/valid.single.ngram.FB5M.torch', wordVocab, relationVocab, 1)
    createRankingData('../Inference/FB5M-ngram/rel.multi.valid.txt',  'data/valid.multi.ngram.FB5M.torch',  wordVocab, relationVocab, 1)

    createRankingData('../Inference/FB5M-heuristics/rel.single.valid.txt', 'data/valid.single.heuristics.FB5M.torch', wordVocab, relationVocab, 1)
    createRankingData('../Inference/FB5M-heuristics/rel.multi.valid.txt',  'data/valid.multi.heuristics.FB5M.torch',  wordVocab, relationVocab, 1)

    createRankingData('../Inference/5M-label/rel.single.test.txt', 'data/test.single.label.FB5M.torch', wordVocab, relationVocab, 1)
    createRankingData('../Inference/5M-label/rel.multi.test.txt',  'data/test.multi.label.FB5M.torch',  wordVocab, relationVocab, 1)

    createRankingData('../Inference/5M-ngram/rel.single.test.txt', 'data/test.single.ngram.FB5M.torch', wordVocab, relationVocab, 1)
    createRankingData('../Inference/5M-ngram/rel.multi.test.txt',  'data/test.multi.ngram.FB5M.torch',  wordVocab, relationVocab, 1)

    createRankingData('../Inference/5M-heuristics/rel.single.test.txt', 'data/test.single.heuristics.FB5M.torch', wordVocab, relationVocab, 1)
    createRankingData('../Inference/5M-heuristics/rel.multi.test.txt',  'data/test.multi.heuristics.FB5M.torch',  wordVocab, relationVocab, 1)

end

trainData()
testData()