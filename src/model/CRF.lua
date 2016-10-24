local CRF, parent = torch.class('CRF', 'nn.Module')

-- initialize the module
function CRF:__init(numClass, maxSeqLen, maxBatch)
    self.numClass  = numClass
    self.maxSeqLen = maxSeqLen
    self.maxBatch  = maxBatch

    -- pairwire parameter
    self.weight     = torch.rand(self.numClass, self.numClass)
    self.gradWeight = torch.zeros(self.numClass, self.numClass)

    -- state memory
    self.alpha = torch.zeros(self.maxSeqLen, self.maxBatch, self.numClass)
    self.beta  = torch.zeros(self.maxSeqLen, self.maxBatch, self.numClass)

    self.partition = torch.zeros(self.maxBatch)

    self.marginalU = torch.zeros(self.maxSeqLen, self.maxBatch, self.numClass)
    self.marginalP = torch.zeros(self.maxSeqLen - 1, self.maxBatch, self.numClass, self.numClass)

    self.output    = torch.Tensor(maxSeqLen, self.maxBatch)
    self.gradInput = torch.Tensor(maxSeqLen, self.maxBatch, self.numClass)

    -- working memory
    self.tempMat = torch.zeros(self.maxBatch, self.numClass, self.numClass)
    self.maxVec  = torch.zeros(self.maxBatch, self.numClass)

    self.uFactor = torch.zeros(self.maxBatch)
    self.pFactor = torch.zeros(self.maxBatch)
    self.flatLabelPair = torch.zeros(self.maxSeqLen, self.maxBatch)

    self.tempGradWeight = torch.zeros(self.maxSeqLen * self.maxBatch, self.numClass * self.numClass)

    -- helper structures
    self.stridePartitionVec = torch.LongStorage({0, 1, 0})
    self.stridePartitionMat = torch.LongStorage({0, 1, 0, 0})
    self.strideWeight = torch.LongStorage({0, 0, self.numClass, 1})

    self.fullVecSize = torch.LongStorage({self.maxSeqLen, self.maxBatch, self.numClass})
    self.fullMatSize = torch.LongStorage({self.maxSeqLen, self.maxBatch, self.numClass, self.numClass})
    self.pairMatSize = torch.LongStorage({self.maxSeqLen - 1, self.maxBatch, self.numClass, self.numClass})
    self.stepMatSize = torch.LongStorage({self.maxBatch, self.numClass, self.numClass})

    -- set training flag 
    self.train = true
end

function CRF:viterbi(input)
    local unary = input
    local seqLen, batchSize = unary:size(1), unary:size(2)

    self.fullMatSize[1], self.fullMatSize[2] = seqLen, batchSize
    self.fullVecSize[1], self.fullVecSize[2] = seqLen, batchSize
    self.stepMatSize[1] = batchSize

    -- resize tensor
    self.alpha:resize(self.fullVecSize):zero()
    self.beta:resize (self.fullVecSize):zero()
    
    self.tempMat:resize(self.stepMatSize)

    -- replicates
    local batchWeight = self.weight:view(1, self.numClass, self.numClass):expand(self.stepMatSize)

    local repUnary = unary:view(seqLen, batchSize, self.numClass, 1):expand(self.fullMatSize)
    local repAlpha = self.alpha:view(seqLen, batchSize, self.numClass, 1):expand(self.fullMatSize)
    local repBeta  = self.beta:view (seqLen, batchSize, self.numClass, 1):expand(self.fullMatSize)

    for i = 1, seqLen do
        self.tempMat:copy(repUnary[i])
        if i ~= seqLen then
            self.tempMat:add(batchWeight)
        end
        if i ~= 1 then
            self.tempMat:add(repAlpha[i-1])
        end
        
        self.alpha[i], self.beta[i] = torch.max(self.tempMat, 2)
    end        

    self.output:resize(seqLen, batchSize, 1):zero()

    self.output[seqLen] = self.beta[{seqLen, {}, 1}]
    for i = seqLen - 1, 1, -1 do
        self.output[i] = self.beta[i]:gather(2, self.output[i+1])
    end

    self.output = self.output:view(seqLen, batchSize)

    return self.output
end

function CRF:forwardbackward(input)
    local unary, label = unpack(input)
    local seqLen, batchSize = unary:size(1), unary:size(2)
    
    self.pairMatSize[1], self.pairMatSize[2] = seqLen - 1, batchSize
    self.fullMatSize[1], self.fullMatSize[2] = seqLen, batchSize
    self.fullVecSize[1], self.fullVecSize[2] = seqLen, batchSize
    self.stepMatSize[1] = batchSize

    -- resize tensor
    self.alpha:resize(self.fullVecSize):zero()
    self.beta:resize(self.fullVecSize):zero()

    self.marginalU:resize(self.fullVecSize)
    self.marginalP:resize(self.pairMatSize)

    self.partition:resize(batchSize)
    
    self.tempMat:resize(self.stepMatSize)
    self.maxVec:resize (batchSize, self.numClass)

    -- replicates
    local fullPartitionVec = self.partition.new(self.partition:storage(), self.partition:storageOffset(), self.fullVecSize, self.stridePartitionVec)
    local pairPartitionMat = self.partition.new(self.partition:storage(), self.partition:storageOffset(), self.pairMatSize, self.stridePartitionMat)

    local pairWeight  = self.weight.new(self.weight:storage(), self.weight:storageOffset(), self.pairMatSize, self.strideWeight)

    local batchWeight = self.weight:view(1, self.numClass, self.numClass):expand(self.stepMatSize)
    local transWeight = batchWeight:transpose(2,3)

    local repUnary  = unary:view(seqLen, batchSize, self.numClass, 1):expand(self.fullMatSize)
    local repAlpha  = self.alpha:view(seqLen, batchSize, self.numClass, 1):expand(self.fullMatSize)
    local repBeta   = self.beta:view (seqLen, batchSize, self.numClass, 1):expand(self.fullMatSize)
    
    local repMaxVec = self.maxVec:view(batchSize, 1, self.numClass):expand(self.stepMatSize)

    -- forward recursion [alpha]
    for i = 1, seqLen do
        self.tempMat:copy(repUnary[i])
        if i ~= seqLen then
            self.tempMat:add(batchWeight)
        end
        if i ~= 1 then
            self.tempMat:add(repAlpha[i-1])
        end

        -- log sum exp
        self.maxVec:max(self.tempMat, 2)
        self.tempMat:add(-1, repMaxVec):exp()
        self.alpha[i]:sum(self.tempMat, 2):log()
        self.alpha[i]:add(self.maxVec)
    end
    
    -- backward recursion [beta]
    for i = seqLen, 1, -1 do
        self.tempMat:copy(repUnary[i])
        if i ~= 1 then
            self.tempMat:add(transWeight)
        end
        if i ~= seqLen then
            self.tempMat:add(repBeta[i+1])
        end

        -- log sum exp
        self.maxVec:max(self.tempMat, 2)
        self.tempMat:add(-1, repMaxVec):exp()
        self.beta[i]:sum(self.tempMat, 2):log()
        self.beta[i]:add(self.maxVec)
    end

    self.partition:copy(self.alpha[{seqLen, {}, 1}])

    -- marginals
    self.marginalU:copy(unary)
    if seqLen >= 2 then
        self.marginalU[{{2, seqLen}}]:add(self.alpha[{{1, seqLen - 1}}])
        self.marginalU[{{1, seqLen - 1}}]:add(self.beta [{{2, seqLen}}])
    end
    self.marginalU:add(-1, fullPartitionVec)
    self.marginalU:exp()
    
    if seqLen >= 2 then
        self.marginalP:add(repUnary[{{1, seqLen - 1}}], repUnary[{{2, seqLen}}]:transpose(3,4))
        self.marginalP:add(pairWeight)
        if seqLen > 2 then
            self.marginalP[{{2, seqLen - 1}}]:add(repAlpha[{{1, seqLen - 2}}])
            self.marginalP[{{1, seqLen - 2}}]:add(repBeta [{{3, seqLen}}]:transpose(3,4))
        end
        self.marginalP:add(-1, pairPartitionMat)
        self.marginalP:exp()
    end    

    -- empirical probability
    self.output:resize(batchSize):zero()
    self.uFactor:resize(batchSize):zero()
    self.pFactor:resize(batchSize):zero()
    self.flatLabelPair:resize(seqLen - 1, batchSize):zero()

    self.uFactor:view(batchSize, 1):sum(unary:view(-1, self.numClass):gather(2, label:view(-1, 1)):view(seqLen, batchSize), 1)

    if seqLen >= 2 then
	    self.flatLabelPair = (label[{{1, seqLen - 1}}] - 1) * self.numClass + label[{{2, seqLen}}]
	    self.pFactor:view(batchSize, 1):sum(self.weight:view(1, -1):gather(2, self.flatLabelPair:view(1, -1)):view(seqLen-1, batchSize), 1)
	end

    self.output:add(self.uFactor, self.pFactor)
    self.output:add(-1, self.partition)
    self.output:exp()
    
    return self.output
end

function CRF:updateOutput(input)
    if self.train then
        return self:forwardbackward(input)
    else
        return self:viterbi(input)
    end
end

function CRF:backward(input)
    local unary, label = unpack(input)
    local seqLen, batchSize = unary:size(1), unary:size(2)

    self.gradInput:resizeAs(unary):zero()
    
    self.gradInput:view(-1, self.numClass):scatter(2, label:view(-1, 1), -1)
    self.gradInput:add(self.marginalU)

    if seqLen >= 2 then
	    self.tempGradWeight:resize((seqLen - 1) * batchSize, self.numClass * self.numClass)
	    
	    self.tempGradWeight:scatter(2, self.flatLabelPair:view(-1, 1), -1)
	    self.tempGradWeight:add(self.marginalP)
	    
	    self.gradWeight:view(1, self.numClass * self.numClass):sum(self.tempGradWeight, 1)
	end

    return self.gradInput
end

function CRF:parameters()
    return {self.weight}, {self.gradWeight}
end