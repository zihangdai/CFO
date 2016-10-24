local BiGRU, parent = torch.class('BiGRU', 'BiRNN')

-- initialize the module
function BiGRU:__init(config)
    parent.__init(self)

    -- config the model
    self.inputSize   = config.inputSize
    self.hiddenSize  = config.hiddenSize
    self.maxSeqLen   = config.maxSeqLen or 200
    self.maxBatch    = config.maxBatch  or 128

    -- allocate weights memory
    self.weight     = torch.Tensor(self.inputSize, self.hiddenSize*6):uniform(-1.0, 1.0)
    self.gradWeight = torch.Tensor(self.inputSize, self.hiddenSize*6):zero()
    
    self.bias     = torch.Tensor(self.hiddenSize*6):uniform(-1.0, 1.0)
    self.gradBias = torch.Tensor(self.hiddenSize*6):zero()

    self.recWeight_G     = torch.Tensor(2, self.hiddenSize, self.hiddenSize*2):uniform(-1.0, 1.0)
    self.gradRecWeight_G = torch.Tensor(2, self.hiddenSize, self.hiddenSize*2):zero()

    self.recWeight_H     = torch.Tensor(2, self.hiddenSize, self.hiddenSize):uniform(-1.0, 1.0)
    self.gradRecWeight_H = torch.Tensor(2, self.hiddenSize, self.hiddenSize):zero()
    
    -- allocate working memory
    self.gates  = torch.Tensor(self.maxSeqLen, self.maxBatch, self.hiddenSize*6):zero()
    self.resetH = torch.Tensor(self.maxSeqLen, self.maxBatch, self.hiddenSize*2):zero()
    self.comple = torch.Tensor(self.maxSeqLen, self.maxBatch, self.hiddenSize*2):zero()
    self.hidden = torch.Tensor(self.maxSeqLen, self.maxBatch, self.hiddenSize*2):zero()

    self.gradGates  = torch.Tensor(self.maxSeqLen, self.maxBatch, self.hiddenSize*6):zero()
    self.gradInput  = torch.Tensor(self.maxSeqLen, self.maxBatch, self.inputSize *2):zero()
    self.gradResetH = torch.Tensor(self.maxSeqLen, self.maxBatch, self.hiddenSize*2):zero()

    self.buffer = torch.ones(self.maxSeqLen*self.maxBatch)

    -- logging information
    if config.logger then
        config.logger.info(string.rep('-', 50))
        config.logger.info('BiGRU Configuration:')
        config.logger.info(string.format('    inputSize   : %5d', self.inputSize))
        config.logger.info(string.format('    hiddenSize  : %5d', self.hiddenSize))
        config.logger.info(string.format('    maxSeqLen   : %5d', self.maxSeqLen))
        config.logger.info(string.format('    maxBatch    : %5d', self.maxBatch))
    end

end

function BiGRU:updateOutput(input)
    assert(self.inputSize==input:size(3), 'Input size not match')
    local seqLen, batchSize = input:size(1), input:size(2)

    self.gates:resize (seqLen, batchSize, self.hiddenSize*6)
    self.resetH:resize(seqLen, batchSize, self.hiddenSize*2)
    self.comple:resize(seqLen, batchSize, self.hiddenSize*2)
    self.hidden:resize(seqLen, batchSize, self.hiddenSize*2)

    self.buffer:resize(seqLen*batchSize)

    self.comple:fill(1)
    
    local denseInput = input:view(seqLen*batchSize, self.inputSize)
    local denseGates = self.gates:view(seqLen*batchSize, self.hiddenSize*6)

    denseGates:addr(0, 1, self.buffer, self.bias)
    denseGates:addmm(1, denseInput, self.weight)

    for i = 1, self.nStream do
        -- set stream: stream 1 deals with forward-GRU & stream 2 deals with backward-GRU
        if cutorch then cutorch.setStream(i) end

        -- get traverse order (depends on the stream)
        local begIdx, endIdx, stride = self:traverseOrder(seqLen, i)

        -- compute stream memory offset
        local left, right = (i-1)*self.hiddenSize, i*self.hiddenSize

        local prevHidden

        for seqIdx = begIdx, endIdx, stride do
            -- get current memory
            local currGates  = self.gates [{seqIdx, {}, {3*left+1, 3*right}}]
            local currResetH = self.resetH[{seqIdx, {}, {  left+1,   right}}]
            local currComple = self.comple[{seqIdx, {}, {  left+1,   right}}]
            local currHidden = self.hidden[{seqIdx, {}, {  left+1,   right}}]

            -- decompose currGates
            local preGateAct = currGates[{{}, {                  1,   self.hiddenSize}}]
            local resetGate  = currGates[{{}, {  self.hiddenSize+1, 2*self.hiddenSize}}]
            local updateGate = currGates[{{}, {2*self.hiddenSize+1, 3*self.hiddenSize}}]
            local bothGates  = currGates[{{}, {  self.hiddenSize+1, 3*self.hiddenSize}}]

            -- recurrent connection
            if seqIdx ~= begIdx then
                bothGates:addmm(1, prevHidden, self.recWeight_G[i])
            end        
            
            -- inplace non-linearity for reset & update (both) gates
            -- bothGates.nn.Sigmoid_forward(bothGates, bothGates)
            bothGates.THNN.Sigmoid_updateOutput(bothGates:cdata(), bothGates:cdata())

            -- reset prev hidden
            if seqIdx ~= begIdx then
                currResetH:cmul(resetGate, prevHidden)
                preGateAct:addmm(1, currResetH, self.recWeight_H[i])
            end
            -- preGateAct.nn.Tanh_forward(preGateAct, preGateAct)
            preGateAct.THNN.Tanh_updateOutput(preGateAct:cdata(), preGateAct:cdata())

            -- complementary gate
            currComple:add(-1, updateGate)

            -- currect hidden
            currHidden:cmul(preGateAct, currComple)
            if seqIdx ~= begIdx then
                currHidden:addcmul(1, prevHidden, updateGate)
            end

            -- set prev hidden
            prevHidden = currHidden
        end
    end

    if cutorch then
        -- set back the stream to default stream (0):
        cutorch.setDefaultStream()

        -- 0 is default stream, let 0 wait for the 2 streams to complete before doing anything further
        cutorch.streamWaitFor(0, self.streamList)
    end

    self.output = self.hidden
    return self.output
end

function BiGRU:updateGradInput(input, gradOutput)
    assert(self.hiddenSize*2==gradOutput:size(gradOutput:nDimension()), 'gradOutput size not match')
    assert(input:size(1)==gradOutput:size(1) and input:size(2)==gradOutput:size(2), 'gradOutput and input size not match')
    
    local seqLen, batchSize = input:size(1), input:size(2)

    self.gradInput:resize (seqLen, batchSize, self.inputSize)
    self.gradGates:resize (seqLen, batchSize, self.hiddenSize*6)
    self.gradResetH:resize(seqLen, batchSize, self.hiddenSize*2)
        
    self.gradGates[1]:fill(0)
    self.gradGates[seqLen]:fill(0)

    for i = 1, self.nStream do
        -- set stream: stream 1 deals with forward-GRU & stream 2 deals with backward-GRU
        if cutorch then cutorch.setStream(i) end

        -- get traverse order (depends on the stream)
        local begIdx, endIdx, stride = self:traverseOrder(seqLen, i)

        -- compute stream memory offset
        local left, right = (i-1)*self.hiddenSize, i*self.hiddenSize

        local prevHidden, prevGradOutput
        
        for seqIdx = endIdx, begIdx, -stride do
            -- get current memory
            local currGates  = self.gates [{seqIdx, {}, {3*left+1, 3*right}}]
            local currResetH = self.resetH[{seqIdx, {}, {  left+1,   right}}]
            local currComple = self.comple[{seqIdx, {}, {  left+1,   right}}]
            local currHidden = self.hidden[{seqIdx, {}, {  left+1,   right}}]

            local currGradGates  = self.gradGates [{seqIdx, {}, {3*left+1, 3*right}}]
            local currGradResetH = self.gradResetH[{seqIdx, {}, {  left+1,   right}}]
            local currGradOutput = gradOutput     [{seqIdx, {}, {  left+1,   right}}]

            -- decompose currGates
            local preGateAct = currGates[{{}, {                  1,   self.hiddenSize}}]
            local resetGate  = currGates[{{}, {  self.hiddenSize+1, 2*self.hiddenSize}}]
            local updateGate = currGates[{{}, {2*self.hiddenSize+1, 3*self.hiddenSize}}]

            local gradPreGateAct = currGradGates[{{}, {                  1,   self.hiddenSize}}]
            local gradResetGate  = currGradGates[{{}, {  self.hiddenSize+1, 2*self.hiddenSize}}]
            local gradUpdateGate = currGradGates[{{}, {2*self.hiddenSize+1, 3*self.hiddenSize}}]
            local gradBothGates  = currGradGates[{{}, {  self.hiddenSize+1, 3*self.hiddenSize}}]

            -- pre-gate input: d_h[t] / d_title{h}[t]
            gradPreGateAct:cmul(currGradOutput, currComple)
            -- gradPreGateAct.nn.Tanh_backward(gradPreGateAct, preGateAct, gradPreGateAct)    -- inplace
            gradPreGateAct.THNN.Tanh_updateGradInput(preGateAct:cdata(), gradPreGateAct:cdata(), gradPreGateAct:cdata(), preGateAct:cdata()) -- inplace

            -- related to prev hidden
            if seqIdx ~= begIdx then
                -- set prev hidden
                prevHidden = self.hidden[{seqIdx-stride, {}, {left+1, right}}]

                -- reset prev hidden: d_h[t] / d_hat{h}[t]
                currGradResetH:mm(gradPreGateAct, self.recWeight_H[i]:t())
                
                -- reset gate: d_h[t] / d_r[t]
                gradResetGate:cmul(currGradResetH, prevHidden)
                -- gradResetGate.nn.Sigmoid_backward(gradResetGate, resetGate, gradResetGate) -- inplace
                gradResetGate.THNN.Sigmoid_updateGradInput(resetGate:cdata(), gradResetGate:cdata(), gradResetGate:cdata(), resetGate:cdata()) -- inplace

                -- update gate: d_h[t] / d_z[t]
                gradUpdateGate:cmul(currGradOutput, prevHidden)
            end

            -- update gate: d_h[t] / d_z[t]
            gradUpdateGate:addcmul(-1, currGradOutput, preGateAct)
            -- gradUpdateGate.nn.Sigmoid_backward(gradUpdateGate, updateGate, gradUpdateGate) -- inplace
            gradUpdateGate.THNN.Sigmoid_updateGradInput(updateGate:cdata(), gradUpdateGate:cdata(), gradUpdateGate:cdata(), updateGate:cdata()) -- inplace

            -- d_h[t] / d_recWeight_H
            self.gradRecWeight_H[i]:addmm(1, currResetH:t(), gradPreGateAct)

            if seqIdx ~= begIdx then
                -- set prev grad hidden/output
                prevGradOutput = gradOutput[{seqIdx-stride, {}, {left+1, right}}]
                
                -- prev hidden: d_h[t] / d_h[t-1]
                prevGradOutput:addmm(1, gradBothGates, self.recWeight_G[i]:t())
                prevGradOutput:addcmul(1, currGradOutput, updateGate)
                prevGradOutput:addcmul(1, currGradResetH, resetGate)

                -- d_h[t] / d_recWeight_G
                self.gradRecWeight_G[i]:addmm(1, prevHidden:t(), gradBothGates)
            end
        end
    end

    if cutorch then
        -- set back the stream to default stream (0):
        cutorch.setDefaultStream()

        -- 0 is default stream, let 0 wait for the 2 streams to complete before doing anything further
        cutorch.streamWaitFor(0, self.streamList)
    end

    local denseInput     = input:view(seqLen*batchSize, self.inputSize)
    local denseGradInput = self.gradInput:view(seqLen*batchSize, self.inputSize)
    local denseGradGates = self.gradGates:view(seqLen*batchSize, self.hiddenSize*6)

    -- d_E / d_input
    denseGradInput:mm(denseGradGates, self.weight:t())

    -- d_E / d_W
    self.gradWeight:addmm(1, denseInput:t(), denseGradGates)

    -- d_E / d_b
    self.gradBias:addmv(1, denseGradGates:t(), self.buffer)

    return self.gradInput
end

function BiGRU:parameters()
    return {self.weight, self.recWeight_G, self.recWeight_H, self.bias}, {self.gradWeight, self.gradRecWeight_G, self.gradRecWeight_H, self.gradBias}
end