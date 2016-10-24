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

   ```python
   cd FocusedLabeling
   th train_crf.lua
   ```

2. Entity Type Vector

   ```python
   cd EntityTypeVec
   th train_ent_typevec.lua
   ```

3. RNN based Relation Network

   ```python
   cd RelationRNN
   th train_rel_rnn.lua
   ```

   ​

