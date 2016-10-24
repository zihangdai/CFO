import sys, os
import cPickle as pickle

def www2fb(in_str):
    out_str = 'fb:%s' % (in_str.split('www.freebase.com/')[-1].replace('/', '.'))
    return out_str

def main():
    in_fn = sys.argv[1]
    db = in_fn.split('-')[-1].split('.')[0]

    out_fn = '%s.core.txt' % (db)
    ent_fn = '%s.ent.pkl' % (db)
    rel_fn = '%s.rel.pkl' % (db)

    ent_dict = {}
    rel_dict = {}
    triple_dict = {}

    with file(in_fn, 'rb') as fi:
        for line in fi:
            fields = line.strip().split('\t')
            sub = www2fb(fields[0])
            rel = www2fb(fields[1])
            objs = fields[2].split()
            if ent_dict.has_key(sub):
                ent_dict[sub] += 1
            else:
                ent_dict[sub] = 1
            if rel_dict.has_key(rel):
                rel_dict[rel] += 1
            else:
                rel_dict[rel] = 1
            for obj in objs:
                obj = www2fb(obj)
                triple_dict[(sub, rel, obj)] = 1
                if ent_dict.has_key(obj):
                    ent_dict[obj] += 1
                else:
                    ent_dict[obj] = 1

    pickle.dump(ent_dict, file(ent_fn, 'wb'))
    with file('%s.ent.txt' % (db), 'wb') as fo:
        for k, v in sorted(ent_dict.items(), key = lambda kv: kv[1], reverse = True):
            print >> fo, k

    pickle.dump(rel_dict, file(rel_fn, 'wb'))
    with file('%s.rel.txt' % (db), 'wb') as fo:
        for k, v in sorted(rel_dict.items(), key = lambda kv: kv[1], reverse = True):
            print >> fo, k

    with file(out_fn, 'wb') as fo:
        for (sub, rel, obj) in triple_dict.keys():
            print >> fo, '<%s>\t<%s>\t<%s>\t.' % (sub, rel, obj)
    print len(triple_dict)

if __name__ == '__main__':
    main()
