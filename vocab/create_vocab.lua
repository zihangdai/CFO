require '..'

function createWordVocab()
    local wordVocab = Vocab('word.glove100k.txt')
    wordVocab:add_unk_token()
    wordVocab:add_pad_token()

    torch.save('vocab.word.t7', wordVocab)
end

function createFBVocab()
    local vocabPath = '../KnowledgeBase'

    local relVocab = Vocab(vocabPath..'/FB5M.rel.txt')
    relVocab:add_unk_token()

    local entVocab = Vocab(vocabPath..'/FB5M.ent.txt')
    entVocab:add_unk_token()

    torch.save('vocab.rel.t7', relVocab)
    torch.save('vocab.ent.t7', entVocab)
end

createWordVocab()
createFBVocab()
