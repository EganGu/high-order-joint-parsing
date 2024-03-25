from __future__ import annotations

from typing import List, Optional, Tuple, Union
import torch
from supar import BiLexicalizedConstituencyCRF
from supar.structs.dist import Semiring, LogSemiring
from supar.utils.fn import diagonal_stripe, expanded_stripe, stripe


class EnhanceLexCRF(BiLexicalizedConstituencyCRF):
    def __init__(self, *args, **kwargs):
        super(EnhanceLexCRF, self).__init__(*args, **kwargs)

    def score(self, value: List[Union[torch.LongTensor, torch.BoolTensor]], partial: bool = False) -> torch.Tensor:
        deps, cons, heads, hooks = value
        s_dep, s_con, s_head, s_hook = self.scores
        mask, lens = self.mask, self.lens
        dep_mask, con_mask = mask[:, 0], mask
        if partial:
            raise NotImplementedError
        s_dep = LogSemiring.prod(LogSemiring.one_mask(s_dep.gather(-1, deps.unsqueeze(-1)).squeeze(-1), ~dep_mask), -1)
        s_head = LogSemiring.mul(s_con, s_head.gather(-1, heads.unsqueeze(-1)).squeeze(-1))
        s_head = LogSemiring.prod(LogSemiring.prod(LogSemiring.one_mask(s_head, ~(con_mask & cons)), -1), -1)
        # cal the scores of hook spans.         
        hook_mask = hooks.ne(-1)
        s_hook = s_hook.gather(-1, hooks.masked_fill(~hook_mask, 0).unsqueeze(-1)).squeeze(-1)
        s_hook = LogSemiring.prod(LogSemiring.prod(LogSemiring.one_mask(s_hook, ~(con_mask & hook_mask)), -1), -1)
        return LogSemiring.mul(s_hook, LogSemiring.mul(s_dep, s_head))

    def forward(self, semiring: Semiring) -> torch.Tensor:
        s_dep, s_con, s_head, s_hook = self.scores
        batch_size, seq_len, *_ = s_con.shape
        # [seq_len, seq_len, batch_size, ...], (m<-h)
        s_dep = semiring.convert(s_dep.movedim(0, 2))
        s_root, s_dep = s_dep[1:, 0], s_dep[1:, 1:]
        # [seq_len, seq_len, batch_size, ...], (i, j)
        s_con = semiring.convert(s_con.movedim(0, 2))
        # [seq_len, seq_len, seq_len-1, batch_size, ...], (i, j, h)
        s_head = semiring.mul(s_con.unsqueeze(2), semiring.convert(s_head.movedim(0, -1)[:, :, 1:]))
        # [seq_len, seq_len, seq_len-1, batch_size, ...], (i, j, h)
        s_span = semiring.zeros_like(s_head)
        # [seq_len, seq_len, seq_len-1, batch_size, ...], (i, j<-h)
        # clone to avoid the inplace err.
        s_hook = semiring.convert(s_hook.movedim(0, -1)[:, :, 1:]).clone()

        diagonal_stripe(s_span, 1).copy_(diagonal_stripe(s_head, 1))
        s_hook.diagonal(1).add_(semiring.mul(s_dep, diagonal_stripe(s_head, 1)).movedim(0, -1))

        for w in range(2, seq_len):
            n = seq_len - w
            # COMPLETE-L: s_span_l(i, j, h) = <s_span(i, k, h), s_hook(h->k, j)>, i < k < j
            # [n, w, batch_size, ...]
            s_l = stripe(semiring.dot(stripe(s_span, n, w-1, (0, 1)), stripe(s_hook, n, w-1, (1, w), False), 1), n, w)
            # COMPLETE-R: s_span_r(i, j, h) = <s_hook(i, k<-h), s_span(k, j, h)>, i < k < j
            # [n, w, batch_size, ...]
            s_r = stripe(semiring.dot(stripe(s_hook, n, w-1, (0, 1)), stripe(s_span, n, w-1, (1, w), False), 1), n, w)
            # COMPLETE: s_span(i, j, h) = (s_span_l(i, j, h) + s_span_r(i, j, h)) * s(i, j, h)
            # [n, w, batch_size, ...]
            s = semiring.mul(semiring.sum(torch.stack((s_l, s_r)), 0), diagonal_stripe(s_head, w))
            diagonal_stripe(s_span, w).copy_(s)

            if w == seq_len - 1:
                continue
            # ATTACH: s_hook(h->i, j) = <s(h->m), s_span(i, j, m)>, i <= m < j
            # [n, seq_len, batch_size, ...]
            s_hook.diagonal(w).add_(semiring.dot(expanded_stripe(s_dep, n, w),
                                                 diagonal_stripe(s_span, w).unsqueeze(2), 1).movedim(0, -1))
        return semiring.unconvert(semiring.dot(s_span[0][self.lens, :, range(batch_size)].transpose(0, 1), s_root, 0))
