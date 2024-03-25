# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Optional, Tuple

import torch
from torch import nn
from supar.modules import Biaffine, Triaffine
from supar.utils.common import MIN
from supar.structs import BiLexicalizedConstituencyCRF, DependencyCRF, ConstituencyCRF
import torch.nn.functional as F
from torch.distributions.utils import lazy_property
from utils.struct import EnhanceLexCRF


def scores_repr(s, word2fencepost=True):
    func = to_fencepost if word2fencepost else to_word
    # only map the span scores.
    return s[0], func(s[1]), *s[2:]


def to_fencepost(s):
    # transform representation from word-based to fencepost-based.
    return s.roll(-1, 1)


def to_word(s):
    return s.roll(1, 1)


def _to_double(*ts: List[torch.Tensor]):
    return [t.double() for t in ts]


def _list_clone(t):
    t_new = []
    for e in t:
        if isinstance(e, torch.Tensor):
            t_new.append(e.clone())
        else:
            t_new.append(e)
    return t_new


def _dict_clone(d):
    d_new = {}
    for k in d.keys():
        if isinstance(d[k], torch.Tensor):
            d_new[k] = d[k].clone()
        else:
            d_new[k] = d[k]
    return d_new


class SemanticsDist(object):
    def __init__(
        self,
        scores,
        lens,
        golds=None,
        tgt='dep'
    ):
        self.scores = scores
        self.golds = golds
        self.lens = lens
        self.tgt = tgt

    @lazy_property
    def dist(self):
        crf = DependencyCRF if self.tgt == 'dep' else ConstituencyCRF
        return crf(self.scores, self.lens)

    @lazy_property
    def marginals(self):
        return self.dist.marginals

    @lazy_property
    def argmax(self):
        return self.dist.argmax

    @lazy_property
    def max(self):
        return self.dist.max

    @lazy_property
    def score(self):
        return self.dist.score(self.golds)

    def neg_likelihoods(self):
        return -self.dist.log_prob(self.golds).sum() / self.lens.sum()

    def max_margin(self, unit: float = 1.):
        return MaxMargin(self.scores.clone(), self.golds, self.lens, unit=unit, tgt=self.tgt).loss


class BiLexicalizedDist(SemanticsDist):
    def __init__(
        self,
        scores: List[torch.Tensor],
        lens: torch.LongTensor,
        golds: Optional[List[torch.LongTensor]] = None
    ) -> BiLexicalizedDist:
        self.raw_scores = scores
        self.raw_golds = golds
        self.lens = lens

    @lazy_property
    def dist(self):
        if self._use_head:
            return EnhanceLexCRF(self._preds, self.lens)
        return BiLexicalizedConstituencyCRF(self._preds, self.lens)

    @lazy_property
    def _use_head(self):
        return len(self.raw_scores) > 2 and self.raw_scores[2] is not None

    @lazy_property
    def _preds(self):
        if self._use_head:
            self.raw_scores = self.raw_scores if self.raw_scores[-1] is not None else \
                (*self.raw_scores[:3],
                 self.raw_scores[-2].new_zeros(self.raw_scores[-2].shape))
            return _to_double(*self.raw_scores)
        else:
            batch_size, seq_len, _ = self.raw_scores[0].shape
            return _to_double(self.raw_scores[0], self.raw_scores[1],
                              self.raw_scores[0].new_zeros((batch_size, seq_len, seq_len, seq_len)))

    @lazy_property
    def _golds(self):
        assert self.raw_golds is not None
        if self._use_head:
            return self.raw_golds
        else:
            batch_size, seq_len, _ = self.raw_scores[0].shape
            return self.raw_golds[0], self.raw_golds[1], self.raw_golds[0].new_zeros((batch_size, seq_len, seq_len)).long()

    @lazy_property
    def marginals(self):
        full_marginals = self.dist.marginals
        if self._use_head:
            return full_marginals
        else:
            return full_marginals[:2]

    @lazy_property
    def score(self):
        return self.dist.score(self._golds)

    def neg_likelihoods(self):
        return -self.dist.log_prob(self._golds).sum() / self.lens.sum()

    def max_margin(self, unit: float = 1., hm=['arc', 'span']):
        return LexicalizedMaxMargin(_list_clone(self.raw_scores), self._golds, self.lens, unit=unit, hm=hm).loss


class MaxMargin(object):
    def __init__(self, preds, golds, lens, unit=1., tgt='dep'):
        self.preds = preds
        self.golds = golds
        self.lens = lens
        self.unit = unit
        self.tgt = tgt

    @lazy_property
    def mask(self):
        _lens = self.lens + 1
        _mask = _lens.unsqueeze(-1).gt(_lens.new_tensor(range(_lens.max())))
        _mask[:, 0] = 0
        return _mask.unsqueeze(1) & _mask.unsqueeze(2)

    def hamming(self):
        if self.tgt == 'dep':
            arc_mask = self.preds.new_zeros(self.preds.shape).bool()
            arc_mask.scatter_(-1, self.golds.unsqueeze(-1), 1)
            arc_mask *= self.mask
            self.preds[arc_mask] -= self.unit
        else:
            self.preds[self.golds] -= self.unit

    @lazy_property
    def loss(self):
        self.hamming()
        lex_dist = SemanticsDist(self.preds, self.lens, self.golds, self.tgt)
        return (lex_dist.max-lex_dist.score).sum() / self.lens.sum()


class LexicalizedMaxMargin(MaxMargin):
    def __init__(
        self,
        preds: List[torch.Tensor],
        golds: List[torch.Tensor],
        lens: torch.LongTensor,
        unit: float = 1.,
        hm: List[str] = ['arc']
    ) -> LexicalizedMaxMargin:
        super().__init__(preds, golds, lens, unit)
        self.hm = hm

    def hamming(self):
        if 'arc' in self.hm:
            arc_mask = self.preds[0].new_zeros(self.preds[0].shape).bool()
            arc_mask.scatter_(-1, self.golds[0].unsqueeze(-1), 1)
            arc_mask *= self.mask
            self.preds[0][arc_mask] -= self.unit
        elif 'span' in self.hm:
            self.preds[1][self.golds[1]] -= self.unit
        elif 'head' in self.hm:
            head_mask = self.preds[2].new_zeros(self.preds[2].shape).bool()
            head_mask.scatter_(-1, self.golds[2].unsqueeze(-1), 1)
            self.preds[2][head_mask] -= self.unit
        else:
            raise KeyError

    @lazy_property
    def loss(self):
        self.hamming()
        lex_dist = BiLexicalizedDist(self.preds, self.lens, self.golds)
        # if ((lex_dist.max-lex_dist.score) < 0).any() and (self.lens < 15).any():
        #     breakpoint()
        # return (lex_dist.max-lex_dist.score) / self.lens
        return (lex_dist.max-lex_dist.score).sum() / self.lens.sum()


class Span2oScorer(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_mlp: int,
        scale: int,
        dropout: float,
        score_type: str = 'biaffine',
        mask: Optional[str] = None
    ):
        super().__init__()
        self.score_type = score_type
        self.mask = mask

        if score_type == 'biaffine':
            self.span2o_attn = Biaffine(
                n_in=n_in, n_proj=n_mlp, scale=scale, bias_x=True, bias_y=True, dropout=dropout)
        elif score_type == 'triaffine':
            self.span2o_attn = Triaffine(
                n_in=n_in, n_proj=n_mlp, scale=scale, bias_x=True, bias_y=True, dropout=dropout)
        else:
            raise ValueError("Score type not supported")

    def __repr__(self):
        return f"{self.span2o_attn}"

    def masked_scores(self, scores):
        _, seq_len = scores.shape[:2]
        mask = scores.new_ones((1, seq_len)).bool()
        mask[:, 0] = 0  # mask the bos
        mask_span = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu(0)
        # (i, j, h)
        mask2o = mask_span.unsqueeze(-1).repeat(1, 1, 1, seq_len)

        ls, rs = torch.stack(torch.where(mask.new_ones(
            seq_len, seq_len))).view(-1, seq_len, seq_len).sort(0)[0]
        mask_bdy = ls.unsqueeze(-1).gt(ls.new_tensor(range(seq_len))
                                       ) | rs.unsqueeze(-1).lt(rs.new_tensor(range(seq_len)))

        if self.mask == 'head':
            mask2o_head = mask2o & ~mask_bdy.unsqueeze(0)
            return scores.masked_fill_(~mask2o_head, MIN)
        elif self.mask == 'hook':
            mask2o_hook = mask2o & mask_bdy.unsqueeze(0)
            return scores.masked_fill_(~mask2o_hook, MIN)
        else:
            raise ValueError

    def forward(self, x, x_span):
        if self.score_type == 'biaffine':
            span_repr = (x_span.unsqueeze(1) - x_span.unsqueeze(2))
            batch_size, seq_len = span_repr.shape[:2]
            span_repr2 = span_repr.reshape(batch_size, seq_len * seq_len, -1)
            # [batch_size, seq_len, seq_len, seq_len] (b, i, j, h)
            s_span2o = self.span2o_attn(span_repr2, x).reshape(
                batch_size, seq_len, seq_len, -1).float()[..., :-1]
        elif self.score_type == 'triaffine':
            s_span2o = self.span2o_attn(x_span, x_span, x)   # (b, h, i, j)
            # [batch_size, seq_len, seq_len, seq_len] (b, i, j, h)
            s_span2o = s_span2o.permute(0, 2, 3, 1).float()

        # if self.mask is not None:
        #     self.masked_scores(s_span2o)

        return s_span2o


class LexMFVI(nn.Module):
    def __init__(
        self,
        max_iter: int = 3,
        structured: bool = False,
        loss: Optional[str] = None
    ) -> LexMFVI:
        super().__init__()
        self.max_iter = max_iter
        self.structured = structured
        self.loss = loss

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter}, structured={self.structured}, loss={self.loss})"

    @torch.enable_grad()
    def forward(
        self,
        scores: List[torch.Tensor],
        features: dict[torch.Tensor],
        masks: List[torch.BoolTensor],
        golds: List[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor]:
        self.lens = masks[0].sum(1)
        scores = {'arc': scores[0], 'span': to_word(
            scores[1]), 'head': scores[2] if len(scores) == 3 else None}
        logits = self.mfvi(scores, features, masks[0])
        return self.marginals_loss(logits, golds, masks)

    def marginals_loss(self, logits, golds, masks):
        logits = scores_repr(logits, True)
        if self.structured:
            dist = BiLexicalizedDist(logits, self.lens, golds)
            marginals = dist.marginals
        else:
            marginals = (logits[0].softmax(-1), logits[1].sigmoid(), logits[2]) if len(logits) > 2 \
                else (logits[0].softmax(-1), logits[1].sigmoid())

        if golds is None:
            return marginals
        else:
            if self.structured:
                loss = dist.neg_likelihoods()
            else:
                if self.loss == 'mm':
                    loss = BiLexicalizedDist(
                        logits, self.lens, golds).max_margin()
                elif self.loss == 'ce':
                    arc_loss = F.cross_entropy(
                        logits[0][masks[0]], golds[0][masks[0]])
                    span_loss = F.binary_cross_entropy_with_logits(
                        logits[1][masks[1]], golds[1][masks[1]].float())
                    loss = arc_loss + span_loss
                    if len(logits) > 2 and logits[2] is not None:
                        # headed span local loss
                        loss += F.cross_entropy(logits[2]
                                                [masks[1]], golds[2][masks[1]])
                else:
                    raise ValueError
            return loss, marginals

    def get_mask(self, mask_dep, m_type='dsp'):
        _, seq_len = mask_dep.shape
        # get index of span left and right boundaries.
        ls, rs = torch.stack(torch.where(mask_dep.new_ones(
            seq_len, seq_len))).view(-1, seq_len, seq_len).sort(0)[0]
        # for word-based span scores, mask the bos token.
        mask_dep_w = mask_dep.index_fill(1, ls.new_tensor(0), 0)
        # for dep scores, the bos token is needed.
        mask_dep = mask_dep.index_fill(1, ls.new_tensor(0), 1)
        if m_type == 'dsp':
            # for word-based scores, the main diagonal represent spans of len 1.
            mask_span = (mask_dep_w.unsqueeze(
                1) & mask_dep_w.unsqueeze(2)).triu(1)
            # [seq_len, seq_len, batch_size], (i, j)
            mask_span = mask_span.movedim(0, 2)
            # pad mask
            mask_pad = mask_dep.unsqueeze(-1) & mask_dep.unsqueeze(-2)
            mask2o_pad = (mask_pad.unsqueeze(
                1) & mask_pad.unsqueeze(2)).permute(1, 2, 3, 0)
            # [seq_len, seq_len, seq_len, batch_size], (i, j, h)
            mask2o_con = mask_span.unsqueeze(2).repeat(1, 1, seq_len, 1)
            mask2o_con = mask2o_con & mask2o_pad
            # h not in [i, j] for [i, j, h]
            mask_bdy = ls.unsqueeze(-1).gt(ls.new_tensor(range(seq_len))
                                           ) | rs.unsqueeze(-1).lt(rs.new_tensor(range(seq_len)))
            mask2o_con = mask2o_con & mask_bdy.unsqueeze(-1)
            # h mask for dep [h, i, j]
            mask2o_dsp = mask2o_con.permute(2, 0, 1, 3)
            return mask2o_dsp

        elif m_type == 'sib':
            # [seq_len, seq_len, batch_size], (h->m)
            mask_dep = (mask_dep.unsqueeze(-1) &
                        mask_dep.unsqueeze(-2)).permute(2, 1, 0)
            # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
            mask2o_dep = mask_dep.unsqueeze(1) & mask_dep.unsqueeze(2)
            mask2o_dep = mask2o_dep & ls.unsqueeze(-1).ne(
                ls.new_tensor(range(seq_len))).unsqueeze(-1)
            mask2o_dep = mask2o_dep & rs.unsqueeze(-1).ne(
                rs.new_tensor(range(seq_len))).unsqueeze(-1)
            return mask2o_dep

        elif m_type == 'pair':
            # get index of span left and right boundaries.
            mask_span = (mask_dep.unsqueeze(1) & mask_dep.unsqueeze(2)).triu(1)
            # [seq_len, seq_len, batch_size], (l->r)
            mask_span = mask_span.movedim(0, 2)
            # [seq_len, seq_len, seq_len, batch_size], (l->r->b)
            mask2o_con = mask_span.unsqueeze(2).repeat(1, 1, seq_len, 1)
            mask_pad = mask_dep.unsqueeze(-1) & mask_dep.unsqueeze(-2)
            mask2o_pad = (mask_pad.unsqueeze(
                1) & mask_pad.unsqueeze(2)).permute(1, 2, 3, 0)

            mask2o_con = mask2o_con & ls.unsqueeze(-1).ne(
                ls.new_tensor(range(seq_len))).unsqueeze(-1)
            mask2o_con = mask2o_con & rs.unsqueeze(-1).ne(
                rs.new_tensor(range(seq_len))).unsqueeze(-1)
            mask2o_con = mask2o_con & mask2o_pad
            return mask2o_con

    def preprocessing_features(self, f, mask):
        mf = {
            'dsp': None, 'sib': None, 'pair': None
        }
        if 'dsp' in f.keys():
            s_dsp = f['dsp']
            mask2o_dsp = self.get_mask(mask, 'dsp')
            if isinstance(s_dsp, tuple) and len(s_dsp) == 2:
                # [seq_len, seq_len, seq_len, batch_size], [h, i, j] (h->i, j) i < j
                s_dsp_l = s_dsp[0].permute(3, 1, 2, 0) * mask2o_dsp
                # [seq_len, seq_len, seq_len, batch_size], [h, i, j] (h->j, i) i < j
                s_dsp_r = s_dsp[1].permute(3, 1, 2, 0) * mask2o_dsp
            else:
                # the upper triangular scores are the dep_span whose head-word is in the left side.
                # And the lower part are the dep_span whose head-word is in the right side.
                s_dsp_l = s_dsp.permute(3, 1, 2, 0) * mask2o_dsp
                s_dsp_r = s_dsp.permute(3, 2, 1, 0) * mask2o_dsp
            mf['dsp'] = (s_dsp_l, s_dsp_r)
        if 'sib' in f.keys():
            # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
            mf['sib'] = f['sib'].permute(
                2, 1, 3, 0) * self.get_mask(mask, 'sib')
        if 'pair' in f.keys():
            # [seq_len, seq_len, seq_len, batch_size], (l, r, b)
            mf['pair'] = f['pair'].permute(
                1, 2, 3, 0) * self.get_mask(mask, 'pair')
        return mf

    def structured_prob(self, s):
        assert self.structured
        marginals = BiLexicalizedDist(
            scores_repr(s, True), self.lens).marginals
        return scores_repr(marginals, False)

    def scores_reshape(self, scores, norm2batch=True):
        dep, con, *_other = scores
        if norm2batch:
            # to [..., batch_size]
            return dep.permute(2, 1, 0), con.permute(1, 2, 0), *_other
        else:
            # to [batch_size, ...]
            return dep.permute(2, 1, 0), con.permute(2, 0, 1), *_other

    def posterior(self, values):
        if self.structured:
            q = self.structured_prob(
                self.scores_reshape((values['arc'], values['span'], values['head']), False))
            q = self.scores_reshape(q, True)
            return {'arc': q[0], 'span': q[1], 'head': q[2] if values['head'] is not None else None}
        else:
            return {'arc': values['arc'].softmax(0), 'span': values['span'].sigmoid(), 'head': values['head']}

    def incorporate(self, q, s, f):
        # headed_span have no ho feature to update.
        m = {'arc': s['arc'], 'span': s['span'], 'head': q['head']}
        if f['dsp'] is not None:
            # use y = y + x rather than y += x to avoid the inplace err.
            m['arc'] = m['arc'] + (q['span'].unsqueeze(0) * f['dsp'][0]
                                   ).sum(2) + (q['span'].unsqueeze(0) * f['dsp'][1]).sum(1)
            m['span'] = m['span'] + (q['arc'].unsqueeze(2) * f['dsp']
                                     [0] + q['arc'].unsqueeze(1) * f['dsp'][1]).sum(0)
        if f['sib'] is not None:
            m['arc'] = m['arc'] + (q['arc'].unsqueeze(1) * f['sib']).sum(2)
        if f['pair'] is not None:
            m['span'] = m['span'] + (q['span'].unsqueeze(1) * f['pair']).sum(2)

        return m

    def mfvi(self, s, features, mask_dep):
        # [seq_len, seq_len, batch_size], (h->i) / (i, j)
        s['arc'], s['span'], s['head'] = self.scores_reshape(
            (s['arc'], s['span'], s['head']), True)
        f = self.preprocessing_features(features, mask_dep)

        # posterior distributions
        # [seq_len, seq_len, batch_size], (h->i) / (i, j)
        q = _dict_clone(s)
        for _ in range(self.max_iter):
            q = self.incorporate(self.posterior(q), s, f)

        return self.scores_reshape((q['arc'], q['span'], q['head']), False)
