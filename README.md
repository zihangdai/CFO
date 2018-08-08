# CFO
Code repo for [Conditional Focused Neural Question Answering with Large-scale Knowledge Bases](https://www.aclweb.org/anthology/P/P16/P16-1076.pdf)

# Installation and Preprocessing
1. Refer to Virtuoso.md to install and confiture the software
2. Make sure [torch7](http://torch.ch/) is installed together with the following dependencies
   - logroll: `luarocks install logroll`
   - nngraph: `luarocks install nngraph`
3. After the installation and configuration of **Virtuoso**, run `bash data_preprocess.sh` to finish preprocessing

# Training

1. Focused Lableing

   ```
   cd FocusedLabeling
   th train_crf.lua
   ```

2. Entity Type Vector

   ```
   cd EntityTypeVec
   th train_ent_typevec.lua
   ```

3. RNN based Relation Network

   ```
   cd RelationRNN
   th train_rel_rnn.lua
   ```

# Inference
In the following, define `SPLIT='valid' or 'test'`.

1. Run focused labeling on validation/test data
   ```
   cd FocusedLabeling
   
   python generate_inference_data.py --split ${SPLIT}
   
   th process_inference.lua
   th infer_crf.lua \
       -testData inference-data/label.${SPLIT}.t7 \
       -modelFile "path-to-pretrained-model"
   ```
   - `python generate_inference_data.py --split ${SPLIT}` will create the file `FocusedLabeling/inference-data/label.${SPLIT}.txt`
   - `th process_inference.lua` will turn the text file `label.${SPLIT}.txt` into `label.${SPLIT}.t7` in torch format (both in the folder `FocusedLabeling/inference-data`)
   - Finally, `th infer_crf.lua ...`  will generate the file `label.result.${SPLIT}` in the folder `FocusedLabeling`.

2. Query candidates based on focused labeling

   ```
   cd Inference
   mkdir ${SPLIT} && cd ${SPLIT}
   python ../query_candidates.py 6 \
          ../../PreprocessData/QAData.${SPLIT}.pkl \
          ../../FocusedLabeling/label.result.${SPLIT} \
          ../../KnowledgeBase/type.top-500.pkl
   ```
   This step will generate the file `QAData.label.${SPLIT}.cpickle` in the folder `Inference/${SPLIT}`.

3. Generate score data based on the query results

   ```
   cd Inference/${SPLIT}
   python ../generate_score_data.py QAData.label.${SPLIT}.cpickle
   ```

   This step will generate the following files in the same folder `Inference/${SPLIT}`:

   - `rel.single.${SPLIT}.txt` (candidate relations for those with only a single candidate subject)
   - `rel.multi.${SPLIT}.txt`   (candidate relations for those with only multiple candidate subject)
   - `type.multi.${SPLIT}.txt` (candidate entities for those with multiple candidate subjects)
   - `single.${SPLIT}.cpickle`
   - `multi.${SPLIT}.cpickle`

4. Run relation inference

   ```
   cd RelationRNN
   mkdir inference-data
   th process_inference.lua -testSplit ${SPLIT}
   th infer_rel_rnn.lua -testData inference-data/rel.single.${SPLIT}.t7
   th infer_rel_rnn.lua -testData inference-data/rel.multi.${SPLIT}.t7
   ```

   This step will generate the files `score.rel.single.${SPLIT}` and `score.rel.multi.${SPLIT}` in the folder `RelationRNN`.

5. Run entity inference

   ```
   cd EntityTypeVec
   mkdir inference-data
   th process_inference.lua -testSplit ${SPLIT}
   th infer_ent_typevec.lua -testData inference-data/ent.${SPLIT}.t7
   ```

   This step will generate the file `score.ent.multi.multi.${SPLIT}` in the folder `EntityTypeVec`.

6. Run joint disambiguation

   ```
   cd Inference/${SPLIT}
   python ../joint_disambiguation.py multi.${SPLIT}.cpickle \
          ../../RelationRNN/score.rel.multi.${SPLIT} \
          ../../EntityTypeVec/score.ent.multi.multi.${SPLIT}
   ```

   