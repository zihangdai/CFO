-- cuda utils
function cudacheck(input)
    if torch.Tensor():type() == 'torch.CudaTensor' then
        input = input:cuda()
    end
    return input
end

function range(b, e)
    local result = cudacheck(torch.LongTensor.range(torch.LongTensor(e-b+1),b,e))
    return result
end

function randperm(up)
    local result = cudacheck(torch.LongTensor.randperm(torch.LongTensor(up),up))
    return result
end

-- loading embedding
function loadPretrainedEmbed (model, embedPath, renorm) 
    local pretrainedEmbed = torch.load(embedPath)
    assert(model.weight:size(2) == pretrainedEmbed:size(2), 'Embedding size does not match')
    model.weight:narrow(1, 1, pretrainedEmbed:size(1)):copy(pretrainedEmbed)
    if renorm then
        model.weight:renorm(2, 2, 1)
    end
end

-- flatten parameters
function flatten(parameters)

    -- returns true if tensor occupies a contiguous region of memory (no holes)
    local function isCompact(tensor)
        local sortedStride, perm = torch.sort(
                torch.LongTensor(tensor:nDimension()):set(tensor:stride()), 1, true)
        local sortedSize = torch.LongTensor(tensor:nDimension()):set(
                tensor:size()):index(1, perm)
        local nRealDim = torch.clamp(sortedStride, 0, 1):sum()
        sortedStride = sortedStride:narrow(1, 1, nRealDim):clone()
        sortedSize    = sortedSize:narrow(1, 1, nRealDim):clone()
        local t = tensor.new():set(tensor:storage(), 1,
                                   sortedSize:storage(),
                                   sortedStride:storage())
        return t:isContiguous()
    end

    if not parameters or #parameters == 0 then
        return torch.Tensor()
    end
    local Tensor = parameters[1].new    

    -- 1. construct the set of all unique storages referenced by parameter tensors
    local storages = {}
    local nParameters = 0
    local parameterMeta = {}
    for k = 1,#parameters do
        local param = parameters[k]
        local storage = parameters[k]:storage()
        local storageKey = torch.pointer(storage)

        if not storages[storageKey] then
            storages[storageKey] = {storage, nParameters}
            nParameters = nParameters + storage:size()
        end

        parameterMeta[k] = {storageOffset = param:storageOffset() +
                                            storages[storageKey][2],
                            size          = param:size(),
                            stride        = param:stride()}
    end

    -- 2. construct a single tensor that will hold all the parameters
    local flatParameters = Tensor(nParameters):zero()

    -- 3. determine if there are elements in the storage that none of the
    --     parameter tensors reference ('holes')
    local tensorsCompact = true
    for k = 1,#parameters do
        local meta = parameterMeta[k]
        local tmp = Tensor():set(
            flatParameters:storage(), meta.storageOffset, meta.size, meta.stride
        )
        tmp:fill(1)
        tensorsCompact = tensorsCompact and isCompact(tmp)
    end

    local maskParameters  = flatParameters:byte():clone()
    local compactOffsets  = flatParameters:long():cumsum(1)
    local nUsedParameters = compactOffsets[-1]

    -- 4. copy storages into the flattened parameter tensor
    for _, storageAndOffset in pairs(storages) do
        local storage, offset = table.unpack(storageAndOffset)
        flatParameters[{{offset+1,offset+storage:size()}}]:copy(Tensor():set(storage))
    end

    -- 5. allow garbage collection
    storages = nil
    for k = 1,#parameters do
        parameters[k]:set(Tensor())
    end

    -- 6. compact the flattened parameters if there were holes
    if nUsedParameters ~= nParameters then
        assert(tensorsCompact, "Cannot gather tensors that are not compact")

        flatParameters = Tensor(nUsedParameters):copy(flatParameters:maskedSelect(maskParameters))
        for k = 1,#parameters do
            parameterMeta[k].storageOffset = compactOffsets[parameterMeta[k].storageOffset]
        end
    end

    -- 7. fix up the parameter tensors to point at the flattened parameters
    for k = 1,#parameters do
        parameters[k]:set(flatParameters:storage(),
            parameterMeta[k].storageOffset,
            parameterMeta[k].size,
            parameterMeta[k].stride)
    end

    return flatParameters
end

-- clone utils
function combineParameters(...)
    --[[ like module:getParameters, but operates on many modules ]]--

    -- get parameters
    local networks = {...}
    local parameters = {}
    local gradParameters = {}
    for i = 1, #networks do
        local net_params, net_grads = networks[i]:parameters()

        if net_params then
            for _, p in pairs(net_params) do
                parameters[#parameters + 1] = p
            end
            for _, g in pairs(net_grads) do
                gradParameters[#gradParameters + 1] = g
            end
        end
    end

    -- flatten parameters and gradients
    local flatParameters = flatten(parameters)
    local flatGradParameters = flatten(gradParameters)

    assert(flatParameters:nElement() == flatGradParameters:nElement(),
        'check that you are sharing parameters and gradParameters')
    if parameters then
        for i = 1, #parameters do
            assert(parameters[i]:storageOffset() == gradParameters[i]:storageOffset(),
               'misaligned parameter at ' .. tostring(i))
        end
    end

    -- return new flat vector that contains all discrete parameters
    return flatParameters, flatGradParameters
end

function cloneManyTimes(net, T)
    local clones = {}
    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end
    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end
    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)
    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone  = reader:readObject()
        reader:close()
        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end
        clones[t] = clone
        collectgarbage()
    end
    mem:close()
    return clones
end
