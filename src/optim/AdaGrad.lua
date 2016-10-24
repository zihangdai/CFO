-- For this AdaGrad implementation, it supports both (optional) traditional momentum 
-- and Nesterov Accelerated Gradient (nag). However, both styles of momentum apply the 
-- same value to all parameters. 

local AdaGrad = torch.class('AdaGrad')

function AdaGrad:__init(gradTab, config)
    self.lr = config.lr
    self.histGradSquare = {}
    for i, grad in pairs(gradTab) do
        self.histGradSquare[i] = grad:clone():fill(1e-4)
    end
    if config.momentum then
        self.momentum = config.momentum        
        self.velocity = {}
        for i, grad in pairs(gradTab) do
            self.velocity[i] = grad:clone():fill(0)
        end
    elseif config.nag then
        self.nag      = config.nag
        self.const_1  = self.nag * self.nag -- NAG from "advances in optimizing recurrent networks"
        self.const_2  = self.nag + 1        -- NAG from "advances in optimizing recurrent networks"
        self.velocity = {}
        for i, grad in pairs(gradTab) do
            self.velocity[i] = grad:clone():fill(0)
        end
    end
    if config.logger then
        config.logger.info(string.rep('-', 50))
        config.logger.info(string.format('AdaGrad Configurations:'))
        for i = 1, #self.lr do
            config.logger.info(string.format('    learning rate [%1d] : %f', i , self.lr[i]))
        end
        if self.momentum then
            config.logger.info(string.format('    classic momentum  : %f', self.momentum))
        elseif self.nag then
            config.logger.info(string.format('    Nesterov momentum : %f', self.nag))
        end
    end
end

function AdaGrad:updateParams(paramsTab, gradTab)
    if self.momentum then
        for i = 1, #paramsTab do
            self.histGradSquare[i]:addcmul(1, gradTab[i], gradTab[i])
            self.velocity[i]:mul(self.momentum):addcdiv(-self.lr[i], gradTab[i], torch.sqrt(self.histGradSquare[i]))
            paramsTab[i]:add(self.velocity[i])
        end
    elseif self.nag then
        for i = 1, #paramsTab do
            self.histGradSquare[i]:addcmul(1, gradTab[i], gradTab[i])
            self.velocity[i]:mul(self.const_1):addcdiv(-self.lr[i]*self.const_2, gradTab[i], torch.sqrt(self.histGradSquare[i]))
            paramsTab[i]:add(self.velocity[i])
        end
    else
        for i = 1, #paramsTab do
            self.histGradSquare[i]:addcmul(1, gradTab[i], gradTab[i])
            paramsTab[i]:addcdiv(-self.lr[i], gradTab[i], torch.sqrt(self.histGradSquare[i]))
        end
    end
end

function AdaGrad:updateMomentum(rate)
    if self.momentum then
        self.momentum = rate
    elseif self.nag then
        self.nag = rate
    end
end

function AdaGrad:effectiveGradNorm(gradTab)
    for i = 1, #gradTab do
        print(string.format('effective norm %d: %f', i, torch.cdiv(gradTab[i], torch.sqrt(self.histGradSquare[i])):norm()))
    end
end
