import sys, os
import io
import cPickle as pickle
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'src/py_module' ))
from QAData import *
import virtuoso

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate_inference_data.py')
    parser.add_argument('--split', default='valid', type=str, help="which data split to consider")
    args = parser.parse_args()
    
    data_list = pickle.load(file('../SimpleQuestions/PreprocessData/QAData.{}.pkl'.format(args.split), 'rb'))
    if not os.path.exists('inference-data'):
        os.mkdir('inference-data')
    
    with io.open('inference-data/label.{}.txt'.format(args.split), 'w', encoding='utf8') as fo:
        for data in data_list:
            if data.text_attention_indices:
                fo.write(u'%s\t%s\n' % (data.question, 
                    ' '.join([str(index) for index in data.text_attention_indices])))
            else:
                fo.write(u'%s\t%s\n' % (data.question, 
                    ' '.join(['0' for _ in data.question.strip().split()])))