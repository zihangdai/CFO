local BatchDot, parent = torch.class('BatchDot', 'nn.Module')

function BatchDot:__init()
    parent.__init(self)
    self.gradInput = {torch.Tensor(), torch.Tensor()}
    self._viewSize = torch.LongStorage()
end 

function BatchDot:updateOutput(input)
    self.output = torch.cmul(input[1], input[2]):sum(input[1]:dim())
    return self.output
end

function BatchDot:updateGradInput(input, gradOutput)
    expandGradOutput  = torch.expand(gradOutput, input[1]:size())
    self.gradInput[1] = torch.cmul(expandGradOutput, input[2])
    self.gradInput[2] = torch.cmul(expandGradOutput, input[1])
    return self.gradInput
end
