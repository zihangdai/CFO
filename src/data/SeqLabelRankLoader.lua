-- file to define the class SeqLabelRankLoader
-- SeqLabelRankLoader:nextBatch() return a batch of 

local SeqLabelRankLoader = torch.class('SeqLabelRankLoader')

function SeqLabelRankLoader:__init(datafile, logger)
    -- sequence & pos match
    local data = torch.load(datafile)
    self.candidates = data.candidates
    self.sequences = data.sequences
    self.posIndex = data.posIndex

    -- additional variables
    self.batchSize = self.sequences[1]:size(2)
    self.numBatch  = #self.sequences
    self.currIdx   = 1
    self.indices   = randperm(self.numBatch)

    if torch.Tensor():type() == 'torch.CudaTensor' then
        for i = 1, self.numBatch do
            self.candidates[i] = self.candidates[i]:cuda()
            self.sequences[i] = self.sequences[i]:cuda()
        end
    end

    if logger then
        self.logger = logger
        self.logger.info(string.rep('-', 50))
        self.logger.info(string.format('SeqLabelRankLoader Configurations:'))
        self.logger.info(string.format('    number of batch : %d', self.numBatch))
        self.logger.info(string.format('    data batch size : %d', self.batchSize))
    end
end

-- sequences[dataIdx]: 2-D LongTensor, [seqLen x batchSize]
-- posIndex[dataIdx]: 2-D LongTensor, [batchSize x numLabel]
function SeqLabelRankLoader:nextBatch(circular)
    if self.currIdx > self.numBatch then
        self.currIdx = 1
        self.indices = randperm(self.numBatch)
    end
    local dataIdx
    if circular then
        dataIdx = self.currIdx
    else
        dataIdx = self.indices[self.currIdx]
    end 
    self.currIdx = self.currIdx + 1

    return self.sequences[dataIdx], self.posIndex[dataIdx], self.candidates[dataIdx]
end

-- create torch-format data for SeqLabelRankLoader
function createSeqLabelRankData(dataPath, savePath, wordVocab, numLabel)
    -- class variables
    local candidates = {}
    local sequences = {}
    local posIndex = {}

    -- read data fileh
    local file = io.open(dataPath, 'r')
    local batchIdx  = 0
    local line
    
    while true do
        line = file:read()
        if line == nil then break end
        batchIdx = batchIdx + 1
        print ('batch '..batchIdx)
        local fields = stringx.split(line, '\t')
        
        -- fields[1]: language sequence
        local tokens = stringx.split(fields[1])
        sequences[batchIdx] = torch.LongTensor(#tokens, 1)

        for i = 1, #tokens do
            local token = tokens[i]
            sequences[batchIdx][{i, 1}] = wordVocab:index(token)
        end
        
        -- fields[2]: correct label
        posIndex[batchIdx] = tonumber(fields[2]) + 1

        -- fields[3:]
        local numCandi = #fields - 2
        candidates[batchIdx] = torch.zeros(numCandi, numLabel)

        for candiIdx = 1, numCandi do
            local labels = stringx.split(fields[candiIdx+2])
            for i = 1, #labels do
                index = tonumber(labels[i]) + 1
                candidates[batchIdx][{candiIdx, index}] = 1
            end
        end

    end
    file:close()

    local data = {}
    data.candidates = candidates
    data.sequences = sequences
    data.posIndex = posIndex

    torch.save(savePath, data)
end
