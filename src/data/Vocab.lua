local Vocab = torch.class('Vocab')

function Vocab:__init(path)
    self.size = 0
    self._index = {}
    self._tokens = {}

    local file = io.open(path)
    while true do
        local line = file:read()
        if line == nil then break end
        self.size = self.size + 1
        self._tokens[self.size] = line
        self._index[line] = self.size
    end
    file:close()

    print('vocab size: '..self.size)
end

function Vocab:contains(w)
    if not self._index[w] then return false end
    return true
end

function Vocab:add(w)
    if self._index[w] ~= nil then
        return self._index[w]
    end
    self.size = self.size + 1
    self._tokens[self.size] = w
    self._index[w] = self.size
    return self.size
end

function Vocab:index(w)
    local index = self._index[w]
    if index == nil then
        if self.unk_index == nil then
            error('Token not in vocabulary and no UNK token defined: ' .. w)
        end
        return self.unk_index
    end
    return index
end

function Vocab:token(i)
    if i < 1 or i > self.size then
        error('Index ' .. i .. ' out of bounds')
    end
    return self._tokens[i]
end

function Vocab:map(tokens)
    local len = #tokens
    local output = torch.IntTensor(len)
    for i = 1, len do
        output[i] = self:index(tokens[i])
    end
    return output
end

function Vocab:add_unk_token()
    if self.unk_token ~= nil then return end
    self.unk_index = self:add('<unk>')
    print('vocab size: '..self.size)
end

function Vocab:add_pad_token()
    if self.pad_token ~= nil then return end
    self.pad_index = self:add('<pad>')
    print('vocab size: '..self.size)
end

function Vocab:add_ent_token()
    if self.ent_token ~= nil then return end
    self.ent_index = self:add('<entity>')
    print('vocab size: '..self.size)
end

function Vocab:add_start_token()
    if self.start_token ~= nil then return end
    self.start_index = self:add('<s>')
    print('vocab size: '..self.size)
end

function Vocab:add_end_token()
    if self.end_token ~= nil then return end
    self.end_index = self:add('</s>')
    print('vocab size: '..self.size)
end

function Vocab:add_space_token()
    if self.space_token ~= nil then return end
    self.space_index = self:add('<_>')
    print('vocab size: '..self.size)
end
