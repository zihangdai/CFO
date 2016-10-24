local BiRNNSelect, parent = torch.class('BiRNNSelect', 'nn.Module')

function BiRNNSelect:__init()
   parent.__init(self)
   self.output    = torch.Tensor()
   self.gradInput = torch.Tensor()
end

function BiRNNSelect:updateOutput(input)
   local seqLen     = input:size(1)
   local batchSize  = input:size(2)
   local doubleSize = input:size(3)
   local hiddenSize = doubleSize / 2

   self.output:resize(batchSize, hiddenSize * 2)

   local fLeft, fRight =            1, hiddenSize
   local bLeft, bRight = hiddenSize+1, doubleSize

   self.output[{{},{fLeft, fRight}}]:copy(input[{{seqLen}, {},{fLeft, fRight}}])
   self.output[{{},{bLeft, bRight}}]:copy(input[{{     1}, {},{bLeft, bRight}}])

   return self.output
end

function BiRNNSelect:updateGradInput(input, gradOutput)
   local seqLen     = input:size(1)
   local doubleSize = input:size(3)
   local hiddenSize = doubleSize / 2

   self.gradInput:resizeAs(input)
   self.gradInput:zero()

   local fLeft, fRight =            1, hiddenSize
   local bLeft, bRight = hiddenSize+1, doubleSize
   
   self.gradInput[{{seqLen}, {},{fLeft, fRight}}]:copy(gradOutput[{{},{fLeft, fRight}}])
   self.gradInput[{{     1}, {},{bLeft, bRight}}]:copy(gradOutput[{{},{bLeft, bRight}}])
   
   return self.gradInput
end 