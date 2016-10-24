local SeqLabelingLoader = torch.class('SeqLabelingLoader')

function SeqLabelingLoader:__init(datafile, logger)
    -- class variables
    local data = torch.load(datafile)
    self.sequences = data.seq
    self.seqLabels = data.label

    -- additional variables
    self.batchSize = self.sequences[1]:size(2)
    self.numBatch = #self.sequences    
    self.currIdx = 1
    self.indices = randperm(self.numBatch)

    if torch.Tensor():type() == 'torch.CudaTensor' then
        for i = 1, self.numBatch do
            self.seqLabels[i]    = self.seqLabels[i]:cuda()
            self.sequences[i] = self.sequences[i]:cuda()
        end
    end

    if logger then
        self.logger = logger
        self.logger.info(string.rep('-', 50))
        self.logger.info(string.format('SeqLabelingLoader Configurations:'))
        self.logger.info(string.format('    number of batch: %d', self.numBatch))
        self.logger.info(string.format('    data batch size: %d', self.batchSize))
    end
end

function SeqLabelingLoader:nextBatch(circular)
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
    return self.sequences[dataIdx], self.seqLabels[dataIdx]
end

-- create torch-format data for SeqLabelingLoader
function createSeqLabelingData(dataPath, savePath, wordVocab, batchSize, noneLabel, trueLabel)
    -- class variable
    local sequences = {}
    local seqLabels = {}

    local noneLabel = noneLabel or 1
    local trueLabel = trueLabel or 2

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

        -- fields[2]: label labels
        local labels = stringx.split(fields[2])

        -- allocate tensor memory
        if seqIdx % batchSize == 0 then
            print('batch: '..batchIdx)
            seqIdx = 1
            batchIdx = batchIdx + 1
            sequences[batchIdx] = torch.LongTensor(#tokens, batchSize):fill(wordVocab.pad_index)
            seqLabels[batchIdx] = torch.DoubleTensor(#tokens, batchSize):fill(noneLabel)
        else
            seqIdx = seqIdx + 1
        end

        -- parse tokens into table
        for i = 1, #tokens do
            sequences[batchIdx][{i, seqIdx}] = wordVocab:index(tokens[i])
        end

        -- parse labels into table
        if #labels == #tokens then
            for i = 1, #labels do
                seqLabels[batchIdx][{i, seqIdx}] = tonumber(labels[i])
            end
        else
            for i = 1, #labels do
                seqLabels[batchIdx][{tonumber(labels[i]) + 1, seqIdx}] = trueLabel
            end
        end
    end
    file:close()

    local data = {}
    data.seq   = sequences
    data.label = seqLabels

    torch.save(savePath, data)
end