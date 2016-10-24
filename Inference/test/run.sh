
predict () {
    echo "predict $1 $2 $3 $4"
    cp $1/score.test.multi.label.FB5M score.multi.valid.FB5M
    cp $2/score.test.label.FB5M score.ent.valid.FB5M
    #python ../joint_predict.py multi.test.cpickle score.multi.valid.FB5M score.ent.valid.FB5M $3 $4
    python ../joint_disambiguation.py multi.test.cpickle score.multi.valid.FB5M score.ent.valid.FB5M
}

predict_symbol () {
    echo "predict symbol $1 $2 $3 $4"
    cp $1/score.test.multi.label.anonymous.FB5M score.multi.valid.FB5M
    cp $2/score.test.label.FB5M score.ent.valid.FB5M
    #python ../joint_predict.py multi.test.cpickle score.multi.valid.FB5M score.ent.valid.FB5M $3 $4
    python ../joint_disambiguation.py multi.test.cpickle score.multi.valid.FB5M score.ent.valid.FB5M
}

predict "../../RelationRNN" "../../EntityTypeVec" 0.85 0.0
predict "../../RelationLTGCNN" "../../EntityTypeVec" 0.85 0.0
predict "../../RelationAverage" "../../EntityTypeVec" 0.85 0.0
predict_symbol "../../RelationLTGCNN" "../../EntityTypeVec" 0.85 0.0

predict "../../RelationRNN" "../../EntityTypeVec" 0.90 0.95
predict "../../RelationLTGCNN" "../../EntityTypeVec" 0.85 0.85
predict "../../RelationAverage" "../../EntityTypeVec" 0.90 0.85
predict_symbol "../../RelationLTGCNN" "../../EntityTypeVec" 0.85 0.85

predict "../../RelationRNN" "../../EntityRNN/TransE" 0.60 0.0
predict "../../RelationLTGCNN" "../../EntityRNN/TransE" 0.55 0.0
predict "../../RelationAverage" "../../EntityRNN/TransE" 0.60 0.0
predict_symbol "../../RelationLTGCNN" "../../EntityRNN/TransE" 0.60 0.0

predict "../../RelationRNN" "../../EntityRNN/TransE" 0.90 0.95
predict "../../RelationLTGCNN" "../../EntityRNN/TransE" 0.50 0.85
predict "../../RelationAverage" "../../EntityRNN/TransE" 0.95 0.95
predict_symbol "../../RelationLTGCNN" "../../EntityRNN/TransE" 0.65 0.95

predict "../../RelationRNN" "../../EntityRNN/Random" 0.75 0.0
predict "../../RelationLTGCNN" "../../EntityRNN/Random" 0.70 0.0
predict "../../RelationAverage" "../../EntityRNN/Random" 0.70 0.0
predict_symbol "../../RelationLTGCNN" "../../EntityRNN/Random" 0.65 0.0

predict "../../RelationRNN" "../../EntityRNN/Random" 0.60 0.95
predict "../../RelationLTGCNN" "../../EntityRNN/Random" 0.70 0.85
predict "../../RelationAverage" "../../EntityRNN/Random" 0.95 0.95
predict_symbol "../../RelationLTGCNN" "../../EntityRNN/Random" 0.95 0.95

predict "../../RelationRNN" "../../EntityAverage"         0.60 0.0
predict "../../RelationLTGCNN" "../../EntityAverage"        0.65 0.0
predict "../../RelationAverage" "../../EntityAverage"                 0.55 0.0
predict_symbol "../../RelationLTGCNN" "../../EntityAverage" 0.65 0.0

predict "../../RelationRNN" "../../EntityAverage"         0.65 0.95
predict "../../RelationLTGCNN" "../../EntityAverage"        0.65 0.85
predict "../../RelationAverage" "../../EntityAverage"                 0.95 0.95
predict_symbol "../../RelationLTGCNN" "../../EntityAverage" 0.65 0.85
