local BiRNN, parent = torch.class('BiRNN', 'nn.Module')

-- initialize the module
function BiRNN:__init(config)
    parent.__init(self)

    -- set cuda streams
    self.nStream = 2
    if cutorch then
        self.streamList = {1, 2}
        if cutorch.getNumStreams() < self.nStream then cutorch.reserveStreams(self.nStream) end
    end
end

function BiRNN:traverseOrder(seqLen, streamIdx)
    if streamIdx == 1 then
        return 1, seqLen, 1
    else
        return seqLen, 1, -1
    end
end

function BiRNN:setAttr(attr, val)
    
end

function BiRNN:evaluate()
    self.train = false
    if cutorch.getNumStreams() < self.nStream then cutorch.reserveStreams(self.nStream) end
end