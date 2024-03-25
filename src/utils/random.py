# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import random
from supar.utils.transform import CoNLL, Tree
from supar.utils.field import Field, RawField


def generate_random_data(data_conll, save_path_conll, n_rand,
                         seed=0, data_tree=None, save_path_tree=None, num=3):
    # split 3 dataset
    def cal_mean_len(data):
        tot_len = 0
        for s in data:
            tot_len += len(s.values[1])
        return tot_len / len(data)

    def chunks(arr, m):
        n = int(math.ceil(len(arr) / float(m)))
        return [arr[i:i + n] for i in range(0, len(arr), n)]
    ARC = Field('arcs', use_vocab=False, fn=CoNLL.get_arcs)
    conll = CoNLL(FORM=(RawField('texts')), HEAD=ARC)

    random.seed(seed)
    print("loading conll data...")
    sc = list(conll.load(data_conll))
    if data_tree is not None:
        print("loading tree data...")
        tree = Tree(TREE=RawField('trees'))
        st = list(tree.load(data_tree))
        assert len(sc) == len(st)

    print("generating the random indexes...")
    sent_idx = random.sample(range(len(sc)), n_rand*num)

    sent_sets = chunks(sent_idx, num)
    for idx, ss in enumerate(sent_sets):
        print(idx)
        print(f"the mean len of whole dataset: {cal_mean_len(sc)}")
        print(
            f"the mean len of selected sub-dataset: {cal_mean_len([sc[i] for i in ss])} ({len(ss)})")

        with open(save_path_conll.split('.')[0]+f'_s{idx}.'+save_path_conll.split('.')[1], 'w') as f:
            for i in ss:
                f.write(str(sc[i]) + '\n')

        if save_path_tree is not None:
            with open(save_path_tree.split('.')[0]+f'_s{idx}'+save_path_tree.split('.')[1], 'w') as f:
                for i in ss:
                    f.write(str(st[i]) + '\n')
