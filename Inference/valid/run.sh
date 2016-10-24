
predict () {
    echo "$1 $2"
    cp $1/score.valid.multi.label.FB5M score.multi.valid.FB5M
    cp $2/score.valid.label.FB5M score.ent.valid.FB5M
    python ../joint_disambiguation.py multi.valid.cpickle score.multi.valid.FB5M score.ent.valid.FB5M
}

predict_symbol () {
    echo "symbol $1 $2"
    cp $1/score.valid.multi.label.anonymous.FB5M score.multi.valid.FB5M
    cp $2/score.valid.label.FB5M score.ent.valid.FB5M
    python ../joint_disambiguation.py multi.valid.cpickle score.multi.valid.FB5M score.ent.valid.FB5M
}

predict "../../RelationRNN" "../../EntityTypeVec"
predict "../../RelationLTGCNN" "../../EntityTypeVec" 
predict "../../RelationAverage" "../../EntityTypeVec"
predict_symbol "../../RelationLTGCNN" "../../EntityTypeVec" 

predict "../../RelationRNN" "../../EntityRNN/TransE"
predict "../../RelationLTGCNN" "../../EntityRNN/TransE" 
predict "../../RelationAverage" "../../EntityRNN/TransE"
predict_symbol "../../RelationLTGCNN" "../../EntityRNN/TransE" 

predict "../../RelationRNN" "../../EntityRNN/Random"
predict "../../RelationLTGCNN" "../../EntityRNN/Random" 
predict "../../RelationAverage" "../../EntityRNN/Random"
predict_symbol "../../RelationLTGCNN" "../../EntityRNN/Random" 

predict "../../RelationRNN" "../../EntityAverage"
predict "../../RelationLTGCNN" "../../EntityAverage"
predict "../../RelationAverage" "../../EntityAverage"
predict_symbol "../../RelationLTGCNN" "../../EntityAverage"
