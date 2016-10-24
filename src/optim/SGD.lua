-- For this SGD implementation, it supports both (optional) traditional momentum 
-- and Nesterov Accelerated Gradient (nag). However, both styles of momentum apply the 
-- same value to all parameters. 

local SGD = torch.class('SGD')

function SGD:__init(gradTab, config)
    self.lr = config.lr
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
    if config.annealing then
        self.annealing = config.annealing
        self.masterLr = {}
        for i = 1, #self.lr do
            self.masterLr[i]  = self.lr[i]
        end
    end
    self.count = 0
    if config.logger then
        config.logger.info(string.rep('-', 50))
        config.logger.info(string.format('SGD Configurations:'))
        for i = 1, #self.lr do
            config.logger.info(string.format('    learning rate [%1d] : %f', i , self.lr[i]))
        end
        if self.momentum then
            config.logger.info(string.format('    classic momentum  : %f', self.momentum))
        elseif self.nag then
            config.logger.info(string.format('    Nesterov momentum : %f', self.nag))
        end
        if self.annealing then
            config.logger.info(string.format('    Annearling rate   : %f', self.annealing))
        end
    end
end

function SGD:updateParams(paramsTab, gradTab)
    self.count = self.count + 1
    if self.annealing then
        for i = 1, #self.masterLr do
            self.lr[i] = self.masterLr[i] / (1 + self.annealing * math.sqrt(self.count))
        end
    end
    -- print (self.lr)
    if self.momentum and self.momentum > 0 then
        for i = 1, #paramsTab do            
            self.velocity[i]:mul(self.momentum):add(-self.lr[i], gradTab[i])
            paramsTab[i]:add(self.velocity[i])
        end
    elseif self.nag and self.nag > 0 then
        for i = 1, #paramsTab do
            self.velocity[i]:mul(self.const_1):add(-self.lr[i]*self.const_2, gradTab[i])
            paramsTab[i]:add(self.velocity[i])
        end
    else
        for i = 1, #paramsTab do
            paramsTab[i]:add(-self.lr[i], gradTab[i])
        end
    end
end

function SGD:updateMomentum(rate)
    if self.momentum then
        self.momentum = rate
    elseif self.nag then
        self.nag = rate
    end
end

function SGD:effectiveGradNorm(gradTab)
    for i = 1, #gradTab do
        print(string.format('effective norm %d: %f', i, gradTab[i]:norm()))
    end
end
