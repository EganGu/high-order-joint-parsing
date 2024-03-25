# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from io import StringIO
from typing import (Iterable, List, Optional, Tuple, Union)
import nltk
from supar.utils.tokenizer import Tokenizer
from torch.distributions.utils import lazy_property
from supar.utils.transform import CoNLL, Tree, Sentence, Batch
from supar.utils.field import Field, RawField
from supar.utils.logging import get_logger
from supar.utils.common import BOS


logger = get_logger(__name__)


class JointTrans(CoNLL, Tree):
    root = ''
    fields = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL',
              'WORD', 'TAG', 'TREE', 'CHART', 'HEADED', 'HOOK']

    def __init__(
        self,
        ID: Optional[Union[Field, Iterable[Field]]] = None,
        FORM: Optional[Union[Field, Iterable[Field]]] = None,
        LEMMA: Optional[Union[Field, Iterable[Field]]] = None,
        CPOS: Optional[Union[Field, Iterable[Field]]] = None,
        POS: Optional[Union[Field, Iterable[Field]]] = None,
        FEATS: Optional[Union[Field, Iterable[Field]]] = None,
        HEAD: Optional[Union[Field, Iterable[Field]]] = None,
        DEPREL: Optional[Union[Field, Iterable[Field]]] = None,
        PHEAD: Optional[Union[Field, Iterable[Field]]] = None,
        PDEPREL: Optional[Union[Field, Iterable[Field]]] = None,
        WORD: Optional[Union[Field, Iterable[Field]]] = None,
        TAG: Optional[Union[Field, Iterable[Field]]] = None,
        TREE: Optional[Union[Field, Iterable[Field]]] = None,
        CHART: Optional[Union[Field, Iterable[Field]]] = None,
        HEADED: Optional[Union[Field, Iterable[Field]]] = None,
        HOOK: Optional[Union[Field, Iterable[Field]]] = None,
        binarize_way: str = ''
    ):
        self.training = True

        # conll property
        self.ID = ID
        self.FORM = FORM
        self.LEMMA = LEMMA
        self.CPOS = CPOS
        self.POS = POS
        self.FEATS = FEATS
        self.HEAD = HEAD
        self.DEPREL = DEPREL
        self.PHEAD = PHEAD
        self.PDEPREL = PDEPREL

        # tree property
        self.WORD = WORD
        self.TAG = TAG
        self.TREE = TREE
        self.CHART = CHART

        # headed span property
        self.HEADED = HEADED
        self.HOOK = HOOK

        self.binarize_way = binarize_way

    @property
    def src(self):
        return self.FORM, self.LEMMA, self.CPOS, self.POS, self.FEATS, self.WORD, self.POS, self.TAG, self.TREE

    @property
    def tgt(self):
        return self.HEAD, self.DEPREL, self.PHEAD, self.PDEPREL, self.CHART, self.HEADED, self.HOOK

    def load(
        self,
        data_dep: Union[str, Iterable],
        data_con: Union[str, Iterable],
        lang: Optional[str] = None,
        proj: bool = False,
        max_len: Optional[int] = None,
        **kwargs
    ) -> Iterable[JointSentence]:
        # conll part
        isconll = False
        if lang is not None:
            tokenizer = Tokenizer(lang)
        if isinstance(data_dep, str) and os.path.exists(data_dep):
            f = open(data_dep)
            if data_dep.endswith('.txt'):
                lines = (i
                         for s in f
                         if len(s) > 1
                         for i in StringIO(self.toconll(s.split() if lang is None else tokenizer(s)) + '\n'))
            else:
                lines, isconll = f, True
        else:
            if lang is not None:
                data_dep = [tokenizer(s) for s in (
                    [data_dep] if isinstance(data_dep, str) else data_dep)]
            else:
                data_dep = [data_dep] if isinstance(
                    data_dep[0], str) else data_dep
            lines = (i for s in data_dep for i in StringIO(
                self.toconll(s) + '\n'))
        # tree part
        if isinstance(data_con, str) and os.path.exists(data_con):
            if data_con.endswith('.txt'):
                data_con = (s.split() if lang is None else tokenizer(s)
                            for s in open(data_con) if len(s) > 1)
            else:
                data_con = open(data_con)
        else:
            if lang is not None:
                data_con = [tokenizer(i) for i in (
                    [data_con] if isinstance(data_con, str) else data_con)]
            else:
                data_con = [data_con] if isinstance(
                    data_con[0], str) else data_con
        data_con = list(data_con)

        index, sentence = 0, []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                tree = nltk.Tree.fromstring(data_con[index]) if isinstance(
                    data_con[index], str) else self.totree(data_con[index], self.root)
                sentence = JointSentence(
                    self, sentence, tree, index, binarize_way=self.binarize_way)
                if isconll and proj and not self.isprojective(list(map(int, sentence.arcs))):
                    logger.warning(
                        f"Sentence {index} is not projective. Discarding it!")
                elif max_len is not None and len(sentence) >= max_len:
                    logger.warning(
                        f"Sentence {index} has {len(sentence)} tokens, exceeding {max_len}. Discarding it!")
                else:
                    yield sentence
                    index += 1
                sentence = []
            else:
                sentence.append(line)


class JointSentence(Sentence):

    def __init__(
        self,
        transform: JointTrans,
        lines: List[str],
        tree: nltk.Tree,
        index: Optional[int] = None,
        binarize_way: str = '',
    ) -> JointSentence:
        super().__init__(transform, index)

        self.values = []
        # record annotations for post-recovery
        self.annotations = dict()

        # conll part (dep)
        for i, line in enumerate(lines):
            value = line.split('\t')
            if value[0].startswith('#') or not value[0].isdigit():
                self.annotations[-i-1] = line
            else:
                self.annotations[len(self.values)] = line
                self.values.append(value)
        self.values = list(zip(*self.values))

        # tree part (con)
        words, tags, chart, head, hook = *zip(*tree.pos()), None, None, None
        # if len(words) == 10 and words[0] == 'Mit' and words[1] == 'Ausnahme':
        #     breakpoint()
        if transform.training:
            chart = [[None]*(len(words)+1) for _ in range(len(words)+1)]
            head = [[0]*(len(words)+1) for _ in range(len(words)+1)]
            hook = [[-1]*(len(words)+1) for _ in range(len(words)+1)]
            trace = [[0]*(len(words)+1) for _ in range(len(words)+1)]

            if binarize_way == 'head':
                arcs = list(map(int, self.arcs))
                offs = [x+1 for x in range(len(arcs))]
                # zero to filter TOP label
                # for German, it contain a lots of trees like (TOP (A x) (B x))
                # which is not consitent with the PTB.
                # If u want to binarize such trees, u can try to delete the zero index
                factors = Tree.factorize(lexically_binarize(tree, arcs)[0])
                # tre = lexically_binarize(tree, arcs)
                for i, j, label in factors:
                    chart[i][j] = label
                    try:
                        head[i][j] = Span(arcs[i:j], offs[i:j], True).head_pos
                    except ValueError:
                        print('wrong head-binarization!')
                        head[i][j] = arcs[i:j][0]
                factors_unlabeled = [x[:2] for x in factors]
                # generate hook span labels.
                for i_l, i_r in factors_unlabeled:
                    if trace[i_l][i_r] == 1:
                        continue
                    for j_l, j_r in factors_unlabeled:
                        if trace[j_l][j_r] == 1:
                            continue
                        if i_r == j_l and (i_l, j_r) in factors_unlabeled:
                            trace[i_l][i_r], trace[j_l][j_r] = 1, 1
                            if head[i_l][i_r] == head[i_l][j_r]:
                                # hook is on right side.
                                try:
                                    hook[j_l][j_r] = Span(
                                        arcs[j_l:j_r], offs[j_l:j_r], True).head_val
                                except ValueError:
                                    hook[j_l][j_r] = arcs[j_l:j_r][0]
                            elif head[j_l][j_r] == head[i_l][j_r]:
                                # hook is on left side.
                                try:
                                    hook[i_l][i_r] = Span(
                                        arcs[i_l:i_r], offs[i_l:i_r], True).head_val
                                except ValueError:
                                    hook[i_l][i_r] = arcs[i_l:i_r][0]
                            else:
                                # wrong cases
                                hook[i_l][i_r] = arcs[i_l:i_r][0]
            elif binarize_way == 'lr':
                for i, j, label in Tree.factorize(Tree.binarize(tree)[0]):
                    chart[i][j] = label
            else:
                raise ValueError("Unknown binarize!")

        self.values += [words, tags, tree, chart, head, hook]

    def __repr__(self):
        # cover the raw lines
        merged = {**self.annotations,
                  **{i: '\t'.join(map(str, line))
                     for i, line in enumerate(zip(*self.values[:-6]))}}
        f_tree = self.values[-4].pformat(1000000)

        return '\n'.join(merged.values()) + '\n' + f_tree + '\n'


class JointBatch(Batch):

    def __init__(
        self,
        sentences: Iterable[Sentence]
    ) -> JointBatch:
        super().__init__(sentences)

    @lazy_property
    def mask_dep(self):
        # ignore the eos
        mask_dep = self.mask[:, 1:]
        # ignore the first token of each sentence
        mask_dep[:, 0] = 0
        return mask_dep

    @lazy_property
    def mask_con(self):
        mask_con = self.mask[:, 1:]
        return (mask_con.unsqueeze(1) & mask_con.unsqueeze(2)).triu_(1)


class Span(object):
    def __init__(
        self,
        arcs: List,
        offset: List,
        headed: bool = False
    ) -> Span:
        self.arcs = tuple(map(int, arcs))
        self.offset = offset
        self.headed = headed

        self.head_val = None
        self.head_pos = None
        if self.headed and self.is_tree():
            self.headed_span()
        else:
            raise ValueError(f"Multi-head Span: {self}\n")

    def mergeable(self, other: Span) -> Span:
        if not self.is_tree() or not other.is_tree():
            return False
        self.update_head()
        other.update_head()
        return self.head_val == other.head_pos or other.head_val == self.head_pos

    def headed_span(self) -> Span:
        if not self.headed:
            self.headed = True
            assert self.is_tree(), 'the span is not tree! '
        self.update_head()
        return self

    def update_head(self):
        if self.headed:
            if 0 in self.arcs:
                self.head_val = 0
                self.head_pos = self.offset[self.arcs.index(0)]
            else:
                find_head = False
                for arc in self.arcs:
                    if arc not in self.offset:
                        if not find_head:
                            find_head = True
                        # elif self.head_val != arc:
                        else:
                            # TODO: there are still some labeled spans are multi-headed.
                            raise ValueError(f"Multi-head Span: {self}\n")
                        self.head_val = arc
                        self.head_pos = self.offset[self.arcs.index(arc)]
            assert (self.head_val is not None) and (self.head_pos is not None), \
                f'{str(self)} Error! '

    def is_tree(self):
        n_root = 0 if 0 not in self.arcs else 1
        for arc in self.arcs:
            if arc not in self.offset and arc != 0:
                n_root += 1
        return n_root == 1

    def add(self, other: Span, headed_check: bool = False) -> Span:
        if headed_check and other.headed and self.headed:
            assert self.mergeable(other)
        self.arcs += other.arcs
        self.offset += other.offset

        if headed_check:
            self.headed_span()
        assert len(self.arcs) == len(self.offset)
        return self

    def __repr__(self) -> str:
        return f'Span {{ arcs: {self.arcs} - offset: {self.offset} - head-val: {self.head_val} - head-pos: {self.head_pos} }}\n'


class Node(object):
    def __init__(self, val) -> None:
        self.val = val
        self.lchildren = []
        self.rchildren = []
        self.visited = False

    def add_child(self, child):
        if child < self.val and child not in self.lchildren:
            self.lchildren.append(child)
        elif child > self.val and child not in self.rchildren:
            self.rchildren.append(child)
        else:
            raise KeyError('Error child value. ')

    def visit(self):
        self.visited = True

    def __repr__(self) -> str:
        return f'Node {{ val: {self.val} - left children: {self.lchildren} - right children: {self.rchildren} }}\n'


def is_proj(arcs):
    # new methods to conduct if a dep tree is projective.
    seq = [i for i in range(len(arcs)+1)]
    tree = [Node(i) for i in seq]
    pairs = [(h, d) for d, h in enumerate(arcs, 1) if h >= 0]

    for h, d in pairs:
        if h == d:
            return False
        tree[h].add_child(d)

    traverse = []

    def track(node: Node):
        for child in node.lchildren:
            track(tree[child])
        if not node.visited:
            traverse.append(node.val)
            node.visit()
        for child in node.rchildren:
            track(tree[child])

    track(tree[0])
    return traverse == seq


def lex_compatible_strict(dep_arcs: List, con_spans: List[Tuple], con_bispans: List[Tuple]) -> bool:
    con_spans = [(t[0], t[1]) for t in con_spans]
    bispans = []
    for sp in con_bispans:
        if (sp[0], sp[1]) not in bispans:
            bispans.append((sp[0], sp[1]))
    seq_len = len(dep_arcs)
    offset = [x for x in range(1, seq_len+1)]

    def track(node) -> Span:
        i, j = next(node)
        headed = (i, j) in con_spans
        if j == i+1:
            return Span([dep_arcs[i:j]], [offset[i:j]], True) if headed else \
                Span([dep_arcs[i:j]], [offset[i:j]])
        else:
            left = track(node)
            right = track(node)

            return left.add(right, headed_check=headed)

    try:
        track(iter(bispans))
        return True
    except AssertionError:
        return False


def filter_lt_trees(data_conll, data_tree, n=5):
    ARC = Field('arcs', use_vocab=False, fn=CoNLL.get_arcs)
    REL = Field('rels', bos=BOS)
    conll, tree = CoNLL(FORM=(RawField('texts')), HEAD=ARC,
                        DEPREL=REL), Tree(TREE=RawField('trees'))
    for sc, st in zip(conll.load(data_conll), tree.load(data_tree)):
        if len(sc.values[1]) == n:
            print(sc.arcs)
            print(sc.rels)
            st.trees.pretty_print()
            # if sc.values[1][0] == 'Some':
            #     import pdb; pdb.set_trace();
            lt = lexically_binarize(st.trees, sc.arcs)
            lt.pretty_print()
            input()


def filter_proj_tree(data_conll, data_tree, save_path_conll='', save_path_tree=''):
    sentences_conll, sentences_tree = [], []
    tot_cnt, lex_cnt, non_lex_cnt = 0, 0, 0

    not_lex_idx = []
    ARC = Field('arcs', use_vocab=False, fn=CoNLL.get_arcs)
    conll, tree = CoNLL(FORM=(RawField('texts')),
                        HEAD=ARC), Tree(TREE=RawField('trees'))

    for sc, st in zip(conll.load(data_conll), tree.load(data_tree)):
        arcs = list(map(int, sc.arcs))
        assert conll.isprojective(arcs) == is_proj(arcs)
        if is_proj(arcs):
            spans = Tree.factorize(st.trees, delete_labels={'TOP'})
            bispans = Tree.factorize(lexically_binarize(
                st.trees, arcs), delete_labels={'TOP'})
            if lex_compatible_strict(arcs, spans, bispans):
                sentences_conll.append(sc)
                sentences_tree.append(st)
                lex_cnt += 1
            else:
                non_lex_cnt += 1
                not_lex_idx.append(tot_cnt)
        else:
            not_lex_idx.append(tot_cnt)

        tot_cnt += 1
        print('\rprocessing: %d (lex) / %d (tot) / %d (non-lex) / %d (non-proj)' %
              (lex_cnt, tot_cnt, non_lex_cnt, tot_cnt-lex_cnt-non_lex_cnt))
        print(len(not_lex_idx), not_lex_idx)

    if save_path_conll != '':
        with open(save_path_conll, 'w') as f:
            f.write('\n'.join([str(i) for i in sentences_conll]) + '\n')
    if save_path_tree != '':
        with open(save_path_tree, 'w') as f:
            f.write('\n'.join([str(i) for i in sentences_tree]) + '\n')


def lexically_binarize(
    tree: nltk.Tree,
    arcs: List[int],
    left: bool = True,
    mark: str = '*',
    join: str = '::',
    implicit: bool = False,
    check_mode: bool = False
) -> nltk.Tree:
    tree = tree.copy(True)
    if check_mode:
        raw_tree = tree.copy(True)
        leaves = []
        for subtree in raw_tree.subtrees():
            if not isinstance(subtree[0], nltk.Tree):
                # add pos tag to make the terminals word unique.
                subtree[0] = f'#{len(leaves)+1}'
                leaves.append(subtree[0])
    # make the unique terminals
    leaves = []
    for subtree in tree.subtrees():
        if not isinstance(subtree[0], nltk.Tree):
            # add pos tag to make the terminals word unique.
            subtree[0] += f'#{len(leaves)}'
            leaves.append(subtree[0])
    offset = [x for x in range(1, len(leaves)+1)]
    boundary = {x: i for i, x in enumerate(leaves)}

    def get_boundary(*subtrees):
        leaves = []
        for st in subtrees:
            leaves += [subtree[0] for subtree in st.subtrees()
                       if not isinstance(subtree[0], nltk.Tree)]
        return boundary[leaves[0]], boundary[leaves[-1]]+1

    nodes = [tree]
    if len(tree) == 1:
        if not isinstance(tree[0][0], nltk.Tree):
            tree[0] = nltk.Tree(f'{tree.label()}{mark}', [tree[0]])
        # make sure the num of children of root is more than one
        nodes = [tree[0]]

    while nodes:
        node = nodes.pop()  # stack
        if isinstance(node, nltk.Tree):  # ignore the terminal node (str format)
            if implicit:
                label = ''
            else:
                label = node.label()
                if mark not in label:
                    label = f'{label}{mark}'
            # ensure that only non-terminals can be attached to a n-ary subtree
            if len(node) > 1:
                for child in node:
                    if not isinstance(child[0], nltk.Tree):
                        # for the pre-terminals, add one more label to the terminal
                        child[:] = [nltk.Tree(child.label(), child[:])]
                        child.set_label(label)
            # chomsky normal form factorization
            if len(node) > 2:
                # from the right side to the left side, which is more similar to left-binarization
                split_range = range(len(node)-2, -1, -1) if left else range(len(node)-1)
                for i in split_range:
                    left, right = get_boundary(
                        *node[:i+1]), get_boundary(*node[i+1:])
                    try:
                        left_span = Span(
                            arcs[left[0]:left[1]], offset[left[0]:left[1]], True)
                        right_span = Span(
                            arcs[right[0]:right[1]], offset[right[0]:right[1]], True)
                    except ValueError:
                        # TODO: how to deal with this issue?
                        continue
                    if left_span.mergeable(right_span):
                        left_len, right_len = len(node[:i+1]), len(node[i+1:])
                        if left_len > 1:
                            node[:i+1] = [nltk.Tree(label, node[:i+1])]
                        if right_len > 1:
                            if left_len > 1:
                                # the lens is changed
                                node[1:] = [nltk.Tree(label, node[1:])]
                            else:
                                node[i+1:] = [nltk.Tree(label, node[i+1:])]
                        break
            # if the tree is not totally lexical
            if len(node) > 2:
                # print("There is one not-lexical data...")
                if left:
                    node[:-1] = [nltk.Tree(label, node[:-1])]
                else:
                    node[1:] = [nltk.Tree(label, node[1:])]
            # add the left subtree and right subtree to the nodes
            nodes.extend(node)
            if check_mode and len(node) == 2:
                left, right = get_boundary(node[0]), get_boundary(node[1])
                try:
                    left_span = Span(arcs[left[0]:left[1]], offset[left[0]:left[1]], True)
                    right_span = Span(arcs[right[0]:right[1]], offset[right[0]:right[1]], True)
                    if not left_span.mergeable(right_span):
                        # print(f'cannot merge {left_span} + {right_span}!!!')
                        return False
                except ValueError as e:
                    # print(e)
                    # raw_tree.pretty_print()
                    # print(arcs)
                    # breakpoint()
                    return False
    if check_mode:
        return True
    # collapse unary productions, should be conducted after binarization
    tree.collapse_unary(joinChar=join)

    # remove the unique label
    for subtree in tree.subtrees():
        if not isinstance(subtree[0], nltk.Tree):
            subtree[0] = subtree[0].split('#')[0]
    return tree
