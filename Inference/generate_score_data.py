import sys
import cPickle as pickle

sys.path.append(os.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'src/py_module' ))
import QAData

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage: python generate_score_data.py cpickle_data'
        sys.exit(-1)
    
    suffix = sys.argv[1].split('.')[-2]
    single_rel_file = file('rel.single.%s.txt'%(suffix), 'wb')
    multi_rel_file  = file('rel.multi.%s.txt'%(suffix), 'wb')
    multi_ent_file  = file('ent.multi.%s.txt'%(suffix), 'wb')
    multi_type_file  = file('type.multi.%s.txt'%(suffix), 'wb')

    data_list = pickle.load(file(sys.argv[1], 'rb'))
    single_rel_data = []
    multi_rel_data  = []
    print >> sys.stderr, 'Finish loading QAData'

    count = 0
    for data in data_list:
        if hasattr(data, 'cand_sub') and hasattr(data, 'cand_rel') and len(data.cand_rel) > 0 and data.relation in data.cand_rel and data.subject in data.cand_sub:
        # if data.subject in data.cand_sub:
            question = data.question
            # Case 1: single candidate subject
            if len(data.cand_sub) == 1:
                print >> single_rel_file, '%s\t%s\t%s' % (question, data.relation, '\t'.join(data.cand_rel))
                single_rel_data.append(data)
            # Case 2: multiple candidate subjects
            elif len(data.cand_sub) > 1:
                print >> multi_rel_file,  '%s\t%s\t%s' % (question, data.relation, '\t'.join(data.cand_rel))
                print >> multi_ent_file,  '%s\t%s\t%s' % (question, data.subject, '\t'.join(data.cand_sub))
                print >> multi_type_file, '%s\t%d\t%s' % (question, data.cand_sub.index(data.subject), '\t'.join([' '.join([str(t) for t in st]) for st in data.sub_types]))
                multi_rel_data.append(data)
        else:
            count += 1

    single_rel_file.close()
    multi_rel_file.close()
    multi_ent_file.close()
    multi_type_file.close()

    pickle.dump(single_rel_data, file('single.%s.cpickle'%(suffix), 'wb'))
    pickle.dump(multi_rel_data, file('multi.%s.cpickle'%(suffix), 'wb'))
    print >> sys.stderr, count
