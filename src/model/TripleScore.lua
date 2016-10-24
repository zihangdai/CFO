function TripleScore(negBatchSize)
    local tarVec = nn.Identity()()
    local posVec = nn.Identity()()
    local negMat = nn.Identity()()
    
    local scoreVecPos = BatchDot() ({tarVec, posVec})
    local scoreMatPos = nn.Replicate(negBatchSize) (scoreVecPos)

    local tarMat      = nn.Replicate(negBatchSize) (tarVec)
    local scoreMatNeg = BatchDot() ({tarMat, negMat})

    return nn.gModule({tarVec, posVec, negMat}, {scoreMatPos, scoreMatNeg})
end
