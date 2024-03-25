from __future__ import annotations

import re
import torch
import numpy as np
from collections import Counter
from typing import List, Optional, Tuple

from supar.utils.metric import Metric, AttachmentMetric, SpanMetric


DELETE = {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}
EQUAL = {'ADVP': 'PRT'}
DELETE_SPMRL = {'TOP', 'ROOT', 'S1', '-NONE-', 'VROOT'}
EQUAL_SPMRL = {}
# the "SYN-xx" label usually to appear in such cases:
# (TOP (S xxx) (SYN-xx xxx))
# There are so few instances of this look (only 2 on train set), but it's certainly not the same style as ptb's.
DELETE_HEBREW = {'TOP', 'ROOT', 'S1', '-NONE-', 'VROOT', "SYN_NN", "SYN_NNP", "SYN_NNT",
                 "SYN_PRP", "SYN_JJ", "SYN_JJT", "SYN_RB", "SYN_RBR", "SYN_MOD", "SYN_VB",
                 "SYN_AUX", "SYN_AGR", "SYN_IN", "SYN_COM", "SYN_REL", "SYN_CC", "SYN_QW",
                 "SYN_HAM", "SYN_WDT", "SYN_DT", "SYN_CD", "SYN_CDT", "SYN_AT", "SYN_H",
                 "SYN_FL", "SYN_ZVL"}
EQUAL_HEBREW = {}


def load_metric_names(metric_type):
    if metric_type == 'joint':
        metric_names = ['LOSS', 'DEP-UCM', 'DEP-LCM', 'UAS', 'LAS',
                        'CON-UCM', 'CON-LCM', 'UP', 'UR', 'UF', 'LP', 'LR', 'LF']
    elif metric_type == 'con':
        metric_names = ['LOSS', 'CON-UCM', 'CON-LCM',
                        'UP', 'UR', 'UF', 'LP', 'LR', 'LF']
    elif metric_type == 'dep':
        metric_names = ['LOSS', 'DEP-UCM', 'DEP-LCM', 'UAS', 'LAS']
    else:
        raise ValueError(f'Unknown metric type: {metric_type}')
    return metric_names


def extract_best_performance(model_dir, out_filepath, metric_type='joint', record_dev=False):
    with open(model_dir+'/model.train.log', 'r') as fr:
        lines = fr.readlines()
    dev_info, test_info = None, None
    for i, line in enumerate(lines):
        if re.search(r'INFO](\s)*Epoch(\s)*[0-9]+(\s)*saved', line) is not None:
            dev_info, test_info = lines[i+1], lines[i+2]
            if test_info.endswith(': \n'):
                test_info = lines[i+3]
            break
    if test_info is not None:
        test_metric = re.findall(r'\d+\.?\d*', test_info)
        metric_names = load_metric_names(metric_type)
        assert len(metric_names) == len(test_metric)

        with open(out_filepath, 'a+') as fa:
            fa.write(f'{model_dir} ')
            for i in range(len(metric_names)):
                fa.write(f'{metric_names[i]}-{test_metric[i]} ')
            fa.write('\n')
    else:
        print('The log file is not completed. ')


def cal_repeated_exp_result(aggregate_file, exp_names, metric_type='joint'):
    agg, exp = {}, []
    with open(aggregate_file, 'r') as fr:
        for line in fr.readlines():
            parts = line.split(' ')
            assert parts[0] not in agg.keys()
            agg[parts[0]] = ' '.join(parts[1:])
    for n in exp_names:
        exp.append(list(map(float, re.findall(r'\d+\.?\d*', agg[n]))))
    exp = np.array(exp)
    metric_names = load_metric_names(metric_type)
    print('Mean metric:')
    mean, std = np.mean(exp, axis=0), np.std(exp, axis=0)
    for i, v in enumerate(mean):
        print(f'{metric_names[i]}: {v:.4f} (±{std[i]:.4f})')


class MultiScore(object):

    def __init__(self, *scores) -> MultiScore:
        self.scores = scores
        self.lens = len(scores)

    def __lt__(self, other: MultiScore) -> bool:
        assert self.lens == other.lens
        rc = True
        for i in range(self.lens):
            if self.scores[i] >= other.scores[i]:
                rc = False
                break
        return rc

    def __le__(self, other: MultiScore) -> bool:
        assert self.lens == other.lens
        rc = True
        for i in range(self.lens):
            if self.scores[i] > other.scores[i]:
                rc = False
                break
        return rc

    def __gt__(self, other: MultiScore) -> bool:
        assert self.lens == other.lens
        rc = True
        for i in range(self.lens):
            if self.scores[i] <= other.scores[i]:
                rc = False
                break
        return rc

    def __ge__(self, other: MultiScore) -> bool:
        assert self.lens == other.lens
        rc = True
        for i in range(self.lens):
            if self.scores[i] < other.scores[i]:
                rc = False
                break
        return rc


class JointMetric(Metric):

    def __init__(
        self,
        loss: Optional[float] = None,
        preds_dep: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        golds_dep: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask_dep: Optional[torch.BoolTensor] = None,
        preds_con: Optional[List[List[Tuple]]] = None,
        golds_con: Optional[List[List[Tuple]]] = None,
        eps: float = 1e-12,
    ) -> JointMetric:
        super().__init__(eps=eps)

        self.total = 0.0
        self.n_ucm_dep = 0.0
        self.n_lcm_dep = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

        self.n_ucm_con = 0.0
        self.n_lcm_con = 0.0
        self.utp = 0.0
        self.ltp = 0.0
        self.pred_con = 0.0
        self.gold_con = 0.0

        self.tt_ucm = 0.0
        self.tt_lcm = 0.0

        if loss is not None:
            self(loss, preds_dep, golds_dep, mask_dep, preds_con, golds_con)

    def __repr__(self):
        s = f"loss: {self.loss:.4f} - "
        # dep
        s += f"Dep: {{ UCM: {self.ucm_dep:6.2%} LCM: {self.lcm_dep:6.2%} "
        s += f"UAS: {self.uas:6.2%} LAS: {self.las:6.2%} }} - "
        # con
        s += f"Con: {{ UCM: {self.ucm_con:6.2%} LCM: {self.lcm_con:6.2%} "
        s += f"UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} "
        s += f"LP: {self.lp:6.2%} LR: {self.lr:6.2%} LF: {self.lf:6.2%} }} "

        s += f"Tot: {{ UCM: {self.tot_ucm:6.2%} LCM: {self.tot_lcm:6.2%}}}"
        return s

    def __call__(
        self,
        loss: float,
        preds_dep: Tuple[torch.Tensor, torch.Tensor],
        golds_dep: Tuple[torch.Tensor, torch.Tensor],
        mask_dep: torch.BoolTensor,
        preds_con: List[List[Tuple]],
        golds_con: List[List[Tuple]],
    ) -> JointMetric:
        lens = mask_dep.sum(1)
        arc_preds, rel_preds, arc_golds, rel_golds = *preds_dep, *golds_dep
        arc_mask = arc_preds.eq(arc_golds) & mask_dep
        rel_mask = rel_preds.eq(rel_golds) & arc_mask
        arc_mask_seq, rel_mask_seq = arc_mask[mask_dep], rel_mask[mask_dep]

        self.n += len(mask_dep)
        self.count += 1
        self.total_loss += float(loss)
        self.n_ucm_dep += arc_mask.sum(1).eq(lens).sum().item()
        self.n_lcm_dep += rel_mask.sum(1).eq(lens).sum().item()

        self.total += len(arc_mask_seq)
        self.correct_arcs += arc_mask_seq.sum().item()
        self.correct_rels += rel_mask_seq.sum().item()

        # span metric
        for i, (pred, gold) in enumerate(zip(preds_con, golds_con)):
            upred, ugold = Counter(
                [tuple(span[:-1]) for span in pred]), Counter([tuple(span[:-1]) for span in gold])
            lpred, lgold = Counter([tuple(span) for span in pred]), Counter(
                [tuple(span) for span in gold])
            utp, ltp = list((upred & ugold).elements()), list(
                (lpred & lgold).elements())
            self.n_ucm_con += len(utp) == len(pred) == len(gold)
            self.n_lcm_con += len(ltp) == len(pred) == len(gold)
            self.utp += len(utp)
            self.ltp += len(ltp)
            self.pred_con += len(pred)
            self.gold_con += len(gold)

            # cal tot match
            span_ucm = len(utp) == len(pred) == len(gold)
            arc_ucm = arc_mask[i, :].sum().eq(lens[i]).item()
            self.tt_ucm += (span_ucm is True) and (arc_ucm is True)

            span_lcm = len(ltp) == len(pred) == len(gold)
            arc_lcm = rel_mask[i, :].sum().eq(lens[i]).item()
            self.tt_lcm += (span_lcm is True) and (arc_lcm is True)

        return self

    def __add__(self, other: JointMetric) -> JointMetric:
        metric = JointMetric(eps=self.eps)
        metric.n = self.n + other.n
        metric.count = self.count + other.count
        metric.total_loss = self.total_loss + other.total_loss

        # dep metric
        metric.n_ucm_dep = self.n_ucm_dep + other.n_ucm_dep
        metric.n_lcm_dep = self.n_lcm_dep + other.n_lcm_dep
        metric.total = self.total + other.total
        metric.correct_arcs = self.correct_arcs + other.correct_arcs
        metric.correct_rels = self.correct_rels + other.correct_rels

        # span metric
        metric.n_ucm_con = self.n_ucm_con + other.n_ucm_con
        metric.n_lcm_con = self.n_lcm_con + other.n_lcm_con
        metric.utp = self.utp + other.utp
        metric.ltp = self.ltp + other.ltp
        metric.pred_con = self.pred_con + other.pred_con
        metric.gold_con = self.gold_con + other.gold_con

        metric.tt_ucm = self.tt_ucm + other.tt_ucm
        metric.tt_lcm = self.tt_lcm + other.tt_lcm

        return metric

    @property
    def score(self):
        # return MultiScore(self.las, self.lf)
        return self.las + self.lf

    @property
    def tot_ucm(self):
        return self.tt_ucm / (self.n + self.eps)

    @property
    def tot_lcm(self):
        return self.tt_lcm / (self.n + self.eps)

    # dep metric
    @property
    def ucm_dep(self):
        return self.n_ucm_dep / (self.n + self.eps)

    @property
    def lcm_dep(self):
        return self.n_lcm_dep / (self.n + self.eps)

    @property
    def uas(self):
        return self.correct_arcs / (self.total + self.eps)

    @property
    def las(self):
        return self.correct_rels / (self.total + self.eps)

    # span metric
    @property
    def ucm_con(self):
        return self.n_ucm_con / (self.n + self.eps)

    @property
    def lcm_con(self):
        return self.n_lcm_con / (self.n + self.eps)

    @property
    def up(self):
        return self.utp / (self.pred_con + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold_con + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred_con + self.gold_con + self.eps)

    @property
    def lp(self):
        return self.ltp / (self.pred_con + self.eps)

    @property
    def lr(self):
        return self.ltp / (self.gold_con + self.eps)

    @property
    def lf(self):
        return 2 * self.ltp / (self.pred_con + self.gold_con + self.eps)

    @property
    def metric_category(self):
        return {
            'loss': ['loss'],

            'dep-full': ['ucm_dep', 'lcm_dep', 'uas', 'las'],
            'con-full': ['ucm_con', 'lcm_con', 'up', 'ur', 'uf', 'lp', 'lr', 'lf'],

            'dep': ['uas', 'las'],
            'con': ['up', 'ur', 'uf', 'lp', 'lr', 'lf'],

        }


def extract_numerical_metric(metric):
    if isinstance(metric, AttachmentMetric):
        return {
            'loss': metric.loss,
            'ucm': metric.ucm,
            'lcm': metric.lcm,
            'uas': metric.uas,
            'las': metric.las,
        }
    elif isinstance(metric, SpanMetric):
        return {
            'loss': metric.loss,
            'ucm': metric.ucm,
            'lcm': metric.lcm,
            'up': metric.up,
            'ur': metric.ur,
            'uf': metric.uf,
            'lp': metric.lp,
            'lr': metric.lr,
            'lf': metric.lf
        }
    elif isinstance(metric, JointMetric):
        return {
            'loss': metric.loss,
            'ucm_dep': metric.ucm_dep,
            'lcm_dep': metric.lcm_dep,
            'uas': metric.uas,
            'las': metric.las,
            'ucm_con': metric.ucm_con,
            'lcm_con': metric.lcm_con,
            'up': metric.up,
            'ur': metric.ur,
            'uf': metric.uf,
            'lp': metric.lp,
            'lr': metric.lr,
            'lf': metric.lf
        }
    else:
        raise TypeError('metric type must in [Attachment, Span, Joint]')
