require '.'

function trainData()
    local wordVocab = torch.load('vocab/vocab.word.t7')
    local entVocab = torch.load('vocab/vocab.ent.t7')
    local relVocab = torch.load('vocab/vocab.rel.t7')

    trainDir = 'SimpleQuestions/trainingData'

    -- focused labeling
    createSeqLabelingData(trainDir..'/data.train.focused_labeling', 'data/train.focused_labeling.t7', wordVocab, 128)

    -- entity network
    createSeqMultiLabelData(trainDir..'/data.train.entity_typevec', 'data/train.entity_typevec.t7', wordVocab, 501, 256)
    
    -- relation network
    createSeqRankingData(trainDir..'/data.train.relation_ranking', 'data/train.relation_ranking.t7', wordVocab, relVocab, 256)
end

trainData()
