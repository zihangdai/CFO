#!/bin/bash
ROOTDIR=`pwd`
KBPATH=${ROOTDIR}/KnowledgeBase/VirtuosoKB/

# 1. download SimpleQuestionv2
echo "====> Step 1: download raw data"
mkdir -p ${ROOTDIR}/RawData
cd ${ROOTDIR}/RawData

wget https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz
tar -xzf SimpleQuestions_v2.tgz

wget https://www.dropbox.com/s/dt4i1a1wayks43n/FB5M-extra.tar.gz
tar -xzf FB5M-extra.tar.gz

# 2. create KB data
echo "====> Step 2: create KB data"
cd ${ROOTDIR}/KnowledgeBase
python convert.py ${ROOTDIR}/RawData/SimpleQuestions_v2/freebase-subsets/freebase-FB5M.txt

mv FB5M.core.txt ${KBPATH}/data/
mv ${ROOTDIR}/RawData/FB5M.*.txt ${KBPATH}/data/

# 3. load data into knowledge base
echo "====> Step 3: load data into knowledge base"
cd ${KBPATH}
./bin/virtuoso-t +foreground +configfile var/lib/virtuoso/db/virtuoso.ini & # start the server
serverPID=$!
sleep 10

./bin/isql 1111 dba dba exec="ld_dir_all('./data', '*', 'fb:');"

pids=()
for i in `seq 1 4`; do
	./bin/isql 1111 dba dba exec="rdf_loader_run();" &
   pids+=($!)
done
for pid in ${pids[@]}; do
     wait $pid
done

# 4. create Vocabs
echo "====> Step 4: create Vocabs"
cd ${ROOTDIR}/vocab
th create_vocab.lua 

5. create training data
ho "====> Step 5: create training data (this will take some time)"

# 5.1. QAData.pkl
cd ${ROOTDIR}/SimpleQuestions/PreprocessData
python process_rawdata.py ${ROOTDIR}/RawData/SimpleQuestions_v2/annotated_fb_data_train.txt 6
python process_rawdata.py ${ROOTDIR}/RawData/SimpleQuestions_v2/annotated_fb_data_valid.txt 6
python process_rawdata.py ${ROOTDIR}/RawData/SimpleQuestions_v2/annotated_fb_data_test.txt 6

# 5.2. create train data in .txt format
cd ${ROOTDIR}/SimpleQuestions
python generate_training_data.py

# 5.3. convert .txt data to .t7 format
cd ${ROOTDIR}
mkdir ${ROOTDIR}/data
th process.lua
