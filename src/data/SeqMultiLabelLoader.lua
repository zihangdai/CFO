local SeqMultiLabelLoader = torch.class('SeqMultiLabelLoader')

function SeqMultiLabelLoader:__init(datafile, logger)
    -- sequence & pos match
    local data = torch.load(datafile)
    self.sequences = data.sequences
    self.seqLabels = data.seqLabels
    if data.seqLength ~= nil then
        self.seqLength = data.seqLength
    end

    -- additional variables
    self.batchSize = self.sequences[1]:size(2)
    self.numBatch  = #self.sequences
    self.currIdx   = 1
    self.indices   = randperm(self.numBatch)

    if torch.Tensor():type() == 'torch.CudaTensor' then
        for i = 1, self.numBatch do
            self.sequences[i] = self.sequences[i]:cuda()
            self.seqLabels[i] = self.seqLabels[i]:cuda()
            if self.seqLength ~= nil then
                self.seqLength[i] = self.seqLength[i]:cuda()
            end
        end
    end

    if logger then
        self.logger = logger
        self.logger.info(string.rep('-', 50))
        self.logger.info(string.format('SeqMultiLabelLoader Configurations:'))
        self.logger.info(string.format('    number of batch : %d', self.numBatch))
        self.logger.info(string.format('    data batch size : %d', self.batchSize))
    end
end

function SeqMultiLabelLoader:nextBatch(circular)
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

    if self.seqLength ~= nil then
        return self.sequences[dataIdx], self.seqLabels[dataIdx], self.seqLength[dataIdx]
    else
        return self.sequences[dataIdx], self.seqLabels[dataIdx]
    end
end

function createSeqMultiLabelData(dataPath, savePath, wordVocab, numLabel, batchSize)
    -- class variables
    local seqLabels = {}
    local sequences = {}
    local seqLength = {}

    -- read data fileh
    local file = io.open(dataPath, 'r')
    local batchIdx = 0    -- the index of sequence batches
    local seqIdx   = 0    -- sequence index within each batch
    local line
    
    while true do
        line = file:read()
        if line == nil then break end
        local fields = stringx.split(line, '\t')
        
        -- fields[1]: language sequence
        local tokens = stringx.split(fields[1])
        -- allocate tensor memory
        if seqIdx % batchSize == 0 then
            print('batch: '..batchIdx)
            seqIdx = 1
            batchIdx = batchIdx + 1
            sequences[batchIdx] = torch.LongTensor(#tokens, batchSize):fill(wordVocab.pad_index)
            seqLength[batchIdx] = torch.LongTensor(batchSize):fill(0)
            seqLabels[batchIdx] = torch.zeros(batchSize, numLabel)
        else
            seqIdx = seqIdx + 1
        end

        -- parse each token in sequence
        for i = 1, #tokens do
            local token = tokens[i]
            sequences[batchIdx][{i, seqIdx}] = wordVocab:index(token)
        end
        seqLength[batchIdx][seqIdx] = #tokens
        
        -- fields[2]: labels
        local labels = stringx.split(fields[2])
        for i = 1, #labels do
            index = tonumber(labels[i]) + 1
            seqLabels[batchIdx][{seqIdx, index}] = 1
        end

    end
    file:close()

    local data = {}
    data.seqLabels = seqLabels
    data.sequences = sequences
    data.seqLength = seqLength

    torch.save(savePath, data)
end
