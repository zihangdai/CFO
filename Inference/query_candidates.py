# -*- coding: utf-8 -*-
import os, sys, re
import multiprocessing as mp
import cPickle as pickle
import numpy as np

sys.path.append(os.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'src/py_module' ))
from QAData import *
import virtuoso
import freebase

type_dict = None
stop_words = ['of', 'on', 'the', 'off', 'in', 'for', 'with', 'a', 'an', 'did', 'does', 'good', 'or', 'not', \
              "'", '?', '!', ':', ',']

def generate_ngrams(tokens, min_len, max_len):
    ngrams = []
    num_token = len(tokens)
    assert(num_token >= max_len)
    for num in range(min_len, max_len+1):
        for i in range(num_token-num+1):
            ngram = ' '.join(tokens[i:i+num])
            if not ngram in stop_words:
                ngrams.append(ngram)
    return list(set(ngrams))

def beg_end_indices(scores, threshold):
    seq_len = len(scores)
    max_idx = np.argmax(scores)
    beg_idx = max_idx
    end_idx = max_idx
    for i in range(max_idx-1,-1,-1):
        if np.abs(scores[i+1] - scores[i]) / scores[i+1] > threshold:
            break
        beg_idx = i 
    for i in range(max_idx+1,seq_len,1):
        if np.abs(scores[i-1] - scores[i]) / scores[i-1] > threshold:
            break
        end_idx = i 
    return beg_idx, end_idx

def form_anonymous_quesion(question, beg_idx, end_idx):
    anonymous_tokens = []
    tokens = question.split()
    anonymous_tokens.extend(tokens[:beg_idx])
    anonymous_tokens.append('X')
    anonymous_tokens.extend(tokens[end_idx+1:])
    anonymous_question = ' '.join(anonymous_tokens)

    return anonymous_question

def query_candidate(data_list, pred_list, pid = 0):
    log_file = open('logs/log.%d.txt'%(pid), 'wb')
    new_data_list = []

    succ_match = 0
    data_index = 0
    for pred, data in zip(pred_list, data_list):
        # incremnt data_index
        data_index += 1

        # extract scores
        scores = [float(score) for score in pred.strip().split()]

        # extract fields needed
        relation = data.relation
        subject  = data.subject
        question = data.question
        tokens   = question.split()
        
        # query name / alias by subject (id)
        candi_sub_list = []
        for threshold in np.arange(0.5, 0.0, -0.095):
            beg_idx, end_idx = beg_end_indices(scores, threshold)
            sub_text = ' '.join(tokens[beg_idx:end_idx+1])
            candi_sub_list.extend(virtuoso.str_query_id(sub_text))
            if len(candi_sub_list) > 0:
                break

        # # using freebase suggest
        # if len(candi_sub_list) == 0:
        #     beg_idx, end_idx = beg_end_indices(scores, 0.2)
        #     sub_text = ' '.join(tokens[beg_idx:end_idx+1])
        #     sub_text = re.sub(r'\s(\w+)\s(n?\'[tsd])\s', r' \1\2 ', sub_text)
        #     suggest_subs = []
        #     for trial in range(3):
        #         try:
        #             suggest_subs = freebase.suggest_id(sub_text)
        #             break
        #         except:
        #             print >> sys.stderr, 'freebase suggest_id error: trial = %d, sub_text = %s' % (trial, sub_text)
        #     candi_sub_list.extend(suggest_subs)
        #     if data.subject not in candi_sub_list:
        #         print >> log_file, '%s\t\t%s\t\t%s\t\t%d' % (sub_text, data.text_subject, fb2www(data.subject), len(candi_sub_list))

        # if potential subject founded
        if len(candi_sub_list) > 0:
            # add candidates to data
            for candi_sub in candi_sub_list:
                candi_rel_list = virtuoso.id_query_out_rel(candi_sub)
                if len(candi_rel_list) > 0:
                    if type_dict:
                        candi_type_list = [type_dict[t] for t in virtuoso.id_query_type(candi_sub) if type_dict.has_key(t)]
                        if len(candi_type_list) == 0:
                            candi_type_list.append(len(type_dict))
                        data.add_candidate(candi_sub, candi_rel_list, candi_type_list)
                    else:
                        data.add_candidate(candi_sub, candi_rel_list)
            data.anonymous_question = form_anonymous_quesion(question, beg_idx, end_idx)
            
            # make score mat
        if hasattr(data, 'cand_sub') and hasattr(data, 'cand_rel'):
            # remove duplicate relations
            data.remove_duplicate()

            # append to new_data_list
            new_data_list.append(data)
                
        # loging information
        if subject in candi_sub_list:
            succ_match += 1

        if data_index % 100 == 0:
            print >> sys.stderr, '[%d] %d / %d' % (pid, data_index, len(data_list))

    print >> log_file, '%d / %d = %f ' % (succ_match, data_index+1, succ_match / float(data_index+1))

    log_file.close()
    pickle.dump(new_data_list, file('temp.%d.cpickle'%(pid),'wb'))

if __name__ == '__main__':
    # Check number of argv
    if len(sys.argv) == 4:
        # Parse input argument
        num_process = int(sys.argv[1])
        data_list   = pickle.load(file(sys.argv[2], 'rb'))
        pred_list   = file(sys.argv[3], 'rb').readlines()
    elif len(sys.argv) == 5:
        # Parse input argument
        num_process = int(sys.argv[1])
        data_list   = pickle.load(file(sys.argv[2], 'rb'))
        pred_list   = file(sys.argv[3], 'rb').readlines()
        type_dict   = pickle.load(file(sys.argv[4], 'rb'))
    else:
        print 'usage: python query_candidate_relation.py num_processes QAData_cpickle_file attention_score_file [[type_dict]]'
        sys.exit(-1)

    suffix = sys.argv[2].split('.')[-2]

    assert(len(data_list) == len(pred_list))

    # Create log directory
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Allocate dataload
    length = len(data_list)
    data_per_p = (length + num_process - 1) / num_process

    # Spawn processes
    processes = [ 
        mp.Process(
            target = query_candidate,
            args = (data_list[pid*data_per_p:(pid+1)*data_per_p], 
                    pred_list[pid*data_per_p:(pid+1)*data_per_p], 
                    pid)   
        )   
        for pid in range(num_process)
    ]  

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    # Merge all data [this will preserve the order]
    new_data_list = []
    for p in range(num_process):
        temp_fn = 'temp.%d.cpickle'%(p)
        new_data_list.extend(pickle.load(file(temp_fn, 'rb')))

    pickle.dump(new_data_list, file('QAData.label.%s.cpickle'%(suffix), 'wb'))
    
    # Remove temp data
    for p in range(num_process):
        temp_fn = 'temp.%d.cpickle'%(p)
        os.remove(temp_fn)
