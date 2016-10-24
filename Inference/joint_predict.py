import os, sys
import glob
import cPickle as pickle
import numpy as np
from sklearn import preprocessing

sys.path.append(os.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'src/py_module' ))
from QAData import *

def predict_func(data, rel_scores, ent_scores, alpha, top_rel_ratio):
    rel_scores = np.array(rel_scores)
    ent_scores = np.array(ent_scores)

    ent_threshold = np.min(ent_scores)
    top_sub_ids = np.where(ent_scores >= ent_threshold)[0]

    rel_threshold = top_rel_ratio * (np.max(rel_scores) - np.min(rel_scores)) + np.min(rel_scores)
    top_rel_ids = np.where(rel_scores >= rel_threshold)[0]

    rel_id_dict = {data.cand_rel[rel_id]:i for i, rel_id in enumerate(top_rel_ids)}

    score_mat = np.zeros((top_sub_ids.shape[0], top_rel_ids.shape[0]))

    for row_idx, sub_id in enumerate(top_sub_ids):
        for rel in data.sub_rels[sub_id]:
            if rel_id_dict.has_key(rel):
                col_idx = rel_id_dict[rel]
                #score_mat[row_idx, col_idx] = rel_scores[top_rel_ids[col_idx]]
                score_mat[row_idx, col_idx] = 1

    # compute all the terms
    ent_scores = ent_scores[top_sub_ids]
    rel_scores = rel_scores[top_rel_ids]
    score_mat = np.exp(score_mat * alpha + ent_scores.reshape(score_mat.shape[0], 1) * (1 - alpha))

    # normalization
    score_mat /= np.sum(score_mat, 0)

    score_mat *= np.exp(rel_scores)

    top_sub_id, top_rel_id = np.unravel_index(np.argmax(score_mat), score_mat.shape)

    return [data.cand_sub[top_sub_ids[top_sub_id]]], data.cand_rel[top_rel_ids[top_rel_id]]

if __name__ == '__main__':
    # Parse input argument
    if len(sys.argv) == 5:
        data_fn = sys.argv[1]
        rel_score_fn = sys.argv[2]  
        ent_score_fn = sys.argv[3]
        alpha = float(sys.argv[4])
        top_rel_ratio = 0.0
    elif len(sys.argv) == 6:
        data_fn = sys.argv[1]
        rel_score_fn = sys.argv[2]
        ent_score_fn = sys.argv[3]
        alpha = float(sys.argv[4])
        top_rel_ratio = float(sys.argv[5])
    else:
        print 'Wrong arguments. Usage: '
        print '  python joint_disambiguation.py cpickle_file rel_score_file ent_score_file alpha [[rel_ratio]]'
        sys.exit(1)

    chosen_subs = 0
    total_subs = 0

    # Error information
    error_dir = './error_analysis'
    if not os.path.exists(error_dir):
        os.makedirs(error_dir)
    category = data_fn.split('.')[0]
    f_0_0 = file(os.path.join(error_dir, 'sub_cor_rel_cor.%s.txt'%(category)), 'wb')
    f_0_1 = file(os.path.join(error_dir, 'sub_cor_rel_err.%s.txt'%(category)), 'wb')
    f_1_0 = file(os.path.join(error_dir, 'sub_err_rel_cor.%s.txt'%(category)), 'wb')
    f_1_1 = file(os.path.join(error_dir, 'sub_err_rel_err.%s.txt'%(category)), 'wb')
    
    # Further disambiguation
    suffix = sys.argv[1].split('.')[-2]

    # Load cPickle file into data
    data_list = pickle.load(file(data_fn, 'rb'))
    print >> sys.stderr, 'finish loading cpickle file %d' % (len(data_list))

    rel_score_list = file(rel_score_fn, 'rb').readlines()
    if ent_score_fn:
        ent_score_list = file(ent_score_fn, 'rb').readlines()
    
    # Count the totol number of data
    corr_mat = np.zeros((2,2))

    for idx, data in enumerate(data_list):
        rel_scores = [float(score) for score in rel_score_list[idx].strip().split(' ')]
        if ent_score_fn:
            ent_scores = [float(score) for score in ent_score_list[idx].strip().split(' ')]
            top_sub, top_rel = predict_func(data, rel_scores, ent_scores, alpha, top_rel_ratio)
        else:
            top_sub, top_rel = rel_based(data, rel_scores)
        
        if len(top_sub) == 1 and top_sub[0] == data.subject:
            if top_rel == data.relation:
                corr_mat[0,0] += 1
                print >> f_0_0, '%s\t%s\t%s\t%s\t%s' % (data.question, fb2www(data.subject), fb2www(top_sub), data.relation, top_rel)
            else:
                corr_mat[0,1] += 1
                print >> f_0_1, '%s\t%s\t%s\t%s\t%s' % (data.question, fb2www(data.subject), fb2www(top_sub), data.relation, top_rel)
        else:
            if top_rel == data.relation:
                corr_mat[1,0] += 1
                print >> f_1_0, '%s\t%s\t%s\t%s\t%s' % (data.question, fb2www(data.subject), fb2www(top_sub), data.relation, top_rel)
            else:
                corr_mat[1,1] += 1
                print >> f_1_1, '%s\t%s\t%s\t%s\t%s' % (data.question, fb2www(data.subject), fb2www(top_sub), data.relation, top_rel)
        

    print alpha
    print corr_mat / len(data_list)
    print corr_mat

    f_0_0.close()
    f_0_1.close()
    f_1_0.close()
    f_1_1.close()
