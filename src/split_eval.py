from supar.utils.transform import CoNLL
from supar.utils.field import Field, RawField
from supar.utils.fn import ispunct


def preprocess_penn2malt(dep_path):
    with open(dep_path) as fp:
        lines = fp.readlines()
    out, idx = [], 0
    for line in lines:
        line = line.strip()
        if len(line):
            wrd, pos, arc, lab = line.split('\t')
            idx += 1
            out.append(f'{idx}\t{wrd}\t_\t{pos}\t_\t_\t{arc}\t{lab}\t_\t_')
        else:
            idx = 0
            out.append('')
    with open(dep_path+'.conll', 'w') as fp:
        fp.write('\n'.join(out) + '\n')


def read_dep(dep_path, tf):
    print(f'Reading {dep_path}...')
    return [{
        'idx': tuple(map(int, sd.values[0])),
        'arc': tuple(map(int, sd.values[6])),
        'wrd': sd.values[1],
        'lab': sd.values[7]
    } for sd in tf.load(dep_path)]


def eval(pred, gold):
    ARC = Field('arcs', use_vocab=False, fn=CoNLL.get_arcs)
    conll = CoNLL(FORM=(RawField('texts')), HEAD=ARC)
    pred = read_dep(pred, conll)
    gold = read_dep(gold, conll)
    n_tot, n_uas, n_las = 0, 0, 0
    for p, g in zip(pred, gold):
        assert len(p['idx']) == len(g['idx'])
        n_tot += sum([not ispunct(w) for w in g['wrd']])
        for pa, pl, ga, gl, w in zip(p['arc'], p['lab'], g['arc'], g['lab'], g['wrd']):
            if ispunct(w):
                continue
            if pa == ga:
                n_uas += 1
                if pl == gl:
                    n_las += 1
    print(f'EVAL: UAS: {n_uas / n_tot: .4f} \tLAS: {n_las / n_tot:.4f}')


def split_joint(joint_path):
    deps, cons = [], []
    with open(joint_path) as fp:
        sent = []
        for line in fp.readlines():
            line = line.strip()
            if len(line):
                sent.append(line)
            else:
                deps.append('\n'.join(sent[:-1])+'\n')
                cons.append(sent[-1])
                sent = []
    tmp_deps = joint_path + '.dep.tmp'
    tmp_cons = joint_path + '.con.tmp'
    for f, s in [(tmp_deps, deps), (tmp_cons, cons)]:
        with open(f, 'w') as fp:
            fp.write('\n'.join(s) + '\n')
    return tmp_deps, tmp_cons
