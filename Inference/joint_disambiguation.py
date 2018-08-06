import os, sys, re
import glob
import cPickle as pickle
import numpy as np
from sklearn import preprocessing

sys.path.append(os.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'src/py_module' ))
from QAData import *

def top_sub_rel(data, rel_scores, ent_scores, alpha, rel_ratio):
    rel_scores = np.array(rel_scores)
    ent_scores = np.array(ent_scores)

    #ent_threshold = np.min(ent_scores)
    #top_sub_ids = np.where(ent_scores >= ent_threshold)[0]
    top_sub_ids = np.arange(ent_scores.shape[0])

    rel_threshold = rel_ratio * (np.max(rel_scores) - np.min(rel_scores)) + np.min(rel_scores)
    top_rel_ids = np.where(rel_scores >= rel_threshold)[0]
    #top_rel_ids = np.arange(rel_scores.shape[0])

    # dict for top relation column idx
    rel_id_dict = {data.cand_rel[rel_id]:i for i, rel_id in enumerate(top_rel_ids)}

    score_mat = np.zeros((top_sub_ids.shape[0], top_rel_ids.shape[0]))
    
    # fill the score matrix
    for row_idx, sub_id in enumerate(top_sub_ids):
        for rel in data.sub_rels[sub_id]:
            if rel_id_dict.has_key(rel):
                col_idx = rel_id_dict[rel]
                #score_mat[row_idx, col_idx] = rel_scores[top_rel_ids[col_idx]]
                score_mat[row_idx, col_idx] = 1

    # compute all the terms
    ent_scores = ent_scores[top_sub_ids]
    rel_scores = rel_scores[top_rel_ids]

    # u(s,r,q) = alpha * I(s->r) + (1 - alpha) * g(q)^T E(s)
    score_mat = np.exp(score_mat * alpha + ent_scores.reshape(score_mat.shape[0], 1) * (1 - alpha))

    # p(s|q,r) propto u(s,r,q)
    score_mat /= np.sum(score_mat, 0)
    
    # p(s|q,r) * p(r|q)
    score_mat *= np.exp(rel_scores)

    #max_score = np.max(score_mat)
    #if np.where(score_mat == max_score)[0].shape[0] > 1:
    #    print np.where(score_mat == max_score)[0].shape[0]

    top_sub_id, top_rel_id = np.unravel_index(np.argmax(score_mat), score_mat.shape)

    return [data.cand_sub[top_sub_ids[top_sub_id]]], data.cand_rel[top_rel_ids[top_rel_id]]

def math(data, rel_scores, ent_scores, alpha = 0.5):
    rel_id_dict = {data.cand_rel[i]:i for i in range(len(data.cand_rel))}
    #rel_scores = preprocessing.scale(rel_scores)
    #ent_scores = preprocessing.scale(ent_scores)
 
    score_mat = np.zeros((len(data.cand_sub), len(data.cand_rel)))
    for i in range(len(data.cand_sub)):
        for rel in data.sub_rels[i]:
            j = rel_id_dict[rel]
            score_mat[i, j] = rel_scores[j]
    
    # compute all the terms
    score_mat = np.exp(score_mat * alpha + np.array(ent_scores).reshape(score_mat.shape[0], 1) * (1 - alpha))

    # normalization
    score_mat /= np.sum(score_mat, 0)

    score_mat *= np.exp(rel_scores)

    top_sub_id, top_rel_id = np.unravel_index(np.argmax(score_mat), score_mat.shape)

    return [data.cand_sub[top_sub_id]], data.cand_rel[top_rel_id]

def weighted_avg(data, rel_scores, ent_scores, alpha = 0.2):
    rel_id_dict = {data.cand_rel[i]:i for i in range(len(data.cand_rel))}
    # rel_scores = preprocessing.scale(rel_scores)
    # ent_scores = preprocessing.scale(ent_scores)

    score_mat = np.zeros((len(data.cand_sub), len(data.cand_rel)))
    for i in range(len(data.cand_sub)):
        for rel in data.sub_rels[i]:
            j = rel_id_dict[rel]
            score_mat[i, j] = rel_scores[j]

    sub_scores = alpha * np.array(ent_scores) + (1 - alpha) * np.sum(score_mat, 1)
    top_sub_score = np.max(sub_scores)
    top_sub_ids = []
    for sub_id in np.argsort(sub_scores)[::-1]:
        if sub_scores[sub_id] < top_sub_score:
            break
        top_sub_ids.append(sub_id)

    top_rel = data.cand_rel[np.argmax(score_mat[top_sub_ids[0]])]
    top_subs = [data.cand_sub[sub_id] for sub_id in top_sub_ids]
    return top_subs, top_rel

def rel_based(data, rel_scores):
    rel_scores = np.array(rel_scores)
    top_rel_ids = np.argsort(rel_scores)
    # rel_scores[top_rel_ids[:-2]] = 0
    # reverse rel->id dict
    rel_id_dict = {data.cand_rel[i]:i for i in range(len(data.cand_rel))}

    score_mat = np.zeros((len(data.cand_sub), len(data.cand_rel)))
    for i in range(len(data.cand_sub)):
        for rel in data.sub_rels[i]:
            j = rel_id_dict[rel]
            score_mat[i, j] = rel_scores[j]

    sub_score = np.sum(score_mat, 1)
    top_subscore = np.max(sub_score)
    top_subid = np.argmax(sub_score)
    top_relid = np.argmax(score_mat[top_subid])

    return [data.cand_sub[top_subid]], data.cand_rel[top_relid]

if __name__ == '__main__':
    # Parse input argument
    if len(sys.argv) == 3:
        data_fn  = sys.argv[1]
        rel_score_fn = sys.argv[2]  
        ent_score_fn = None
    elif len(sys.argv) == 4:
        data_fn  = sys.argv[1]
        rel_score_fn = sys.argv[2]  
        ent_score_fn = sys.argv[3]
    else:
        print 'Wrong arguments. Usage: '
        print '  python joint_disambiguation.py cpickle_file rel_score_file ent_score_file'
        sys.exit(1)

    chosen_subs = 0
    total_subs = 0

    count_multi = 0

    # Error information
    error_dir = './error_analysis'
    if not os.path.exists(error_dir):
        os.makedirs(error_dir)
    category = data_fn.split('.')[0]
    
    # Load cPickle file into data
    data_list = pickle.load(file(data_fn, 'rb'))
    print >> sys.stderr, 'finish loading cpickle file %d' % (len(data_list))

    rel_score_list = file(rel_score_fn, 'rb').readlines()
    if ent_score_fn:
        ent_score_list = file(ent_score_fn, 'rb').readlines()
    
    # Count the totol number of data
    for rel_ratio in [0, 0.75, 0.85, 0.95]:
    #for rel_ratio in [0]:
        print '=' * 120
        #for alpha in np.arange(0.05,1.00,0.05):
        for alpha in np.arange(0.05,1.01,0.05):
            # Rescore for each data in data_list
            corr_mat = np.zeros((2,2))

            count = 0
            for idx, data in enumerate(data_list):
                rel_scores = [float(score) for score in rel_score_list[idx].strip().split(' ')]
                ent_scores = [float(score) for score in ent_score_list[idx].strip().split(' ')]
                # top_sub, top_rel = rel_based(data, rel_scores)
                # top_sub, top_rel = weighted_avg(data, rel_scores, ent_scores)
                # top_sub, top_rel = math(data, rel_scores, ent_scores, alpha)
                top_sub, top_rel = top_sub_rel(data, rel_scores, ent_scores, alpha, rel_ratio)
                
                if len(top_sub) == 1 and top_sub[0] == data.subject:
                    if top_rel == data.relation:
                        corr_mat[0,0] += 1
                    else:
                        corr_mat[0,1] += 1
                else:
                    if top_rel == data.relation:
                        corr_mat[1,0] += 1
                    else:
                        corr_mat[1,1] += 1
                
            print '%4.3f, %4.3f, %d' % (alpha, rel_ratio, corr_mat[0,0])
