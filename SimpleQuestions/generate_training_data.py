import sys, os
import io
import cPickle as pickle

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'src/py_module' ))
from QAData import *
import virtuoso

def focused_labeling_data(data_list):
    with io.open('trainingData/data.train.focused_labeling', 'w', encoding='utf8') as fo:
        for data in data_list:
            if data.text_attention_indices:
                fo.write(u'%s\t%s\n' % (data.question, ' '.join([str(index) for index in data.text_attention_indices])))

def relation_ranking_data(data_list):
    fo = io.open('trainingData/data.train.relation_ranking', 'w', encoding='utf8')

    # Main Loop
    data_turple = []
    data_num = 0
    for data in data_list:
        question = data.question
        pos_rel  = data.relation
        
        # this condition will filter out any question that has only one word
        if len(question.split()) > 1:
            data_turple.append((question, pos_rel))
            data_num += 1

    # will choose to output data according to indices
    chosen_num = data_num - (data_num % 256)
    chosen_indices = np.sort(np.random.permutation(data_num)[:chosen_num])

    chosen_indices_idx = 0
    # for each data triple in data_turple list
    for idx in range(len(data_turple)):
        question = data_turple[idx][0]
        pos_rel  = data_turple[idx][1]
        if idx == chosen_indices[chosen_indices_idx]:
            fo.write(u'%s\t%s\n' % (question, pos_rel))
            chosen_indices_idx += 1

    fo.close()

def entity_ranking_data(data_list):
    fo = io.open('trainingData/data.train.entity_ranking', 'w', encoding='utf8')

    # Main Loop
    data_turple = []
    data_num = 0
    for data in data_list:
        pos_sub  = data.subject
        pos_rel  = data.relation
        question = data.question
        
        # this condition will filter out any question that has only one word
        if len(question.split()) > 1:
            data_turple.append((question, pos_sub, pos_rel))
            data_num += 1

    # will choose to output data according to indices
    chosen_num = data_num - (data_num % 256)
    chosen_indices = np.sort(np.random.permutation(data_num)[:chosen_num])

    chosen_indices_idx = 0
    # for each data triple in data_turple list
    for idx in range(len(data_turple)):
        question = data_turple[idx][0]
        pos_sub  = data_turple[idx][1]
        pos_rel  = data_turple[idx][2]
        if idx == chosen_indices[chosen_indices_idx]:
            fo.write(u'%s\t%s\t%s\n' % (question, pos_sub, pos_rel))
            chosen_indices_idx += 1

    fo.close()

def entity_typevec_data(data_list):
    type_dict = pickle.load(file('../KnowledgeBase/type.top-500.pkl', 'rb'))
    with io.open('trainingData/data.train.entity_typevec', 'w', encoding='utf8') as fo:
        for data in data_list:
            sub = data.subject
            question = data.question
            types = virtuoso.id_query_type(sub)
            types = [t for t in types if type_dict.has_key(t)]
            if len(types) > 0:
                fo.write(u'%s\t%s\n' % (question, ' '.join([str(type_dict[t]) for t in types])))
            else:
                fo.write(u'%s\t%d\n' % (question, len(type_dict)))


if __name__ == '__main__':
    data_list = pickle.load(file('PreprocessData/QAData.train.pkl', 'rb'))
    if not os.path.exists('trainingData'):
        os.mkdir('trainingData')
    print >> sys.stderr, 'focused_labeling_data'
    focused_labeling_data(data_list)
    print >> sys.stderr, 'relation_ranking_data'
    relation_ranking_data(data_list)
    # print >> sys.stderr, 'entity_ranking_data'
    # entity_ranking_data(data_list)
    print >> sys.stderr, 'entity_typevec_data'
    entity_typevec_data(data_list)
