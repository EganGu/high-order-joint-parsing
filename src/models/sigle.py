import torch
from supar.models.dep import CRFDependencyModel
from supar.models.const import CRFConstituencyModel
from utils.modules import SemanticsDist
from utils.modules import to_fencepost, Span2oScorer, BiLexicalizedDist, LexMFVI


class MaxMarginDepModel(CRFDependencyModel):
    def loss(self, s_arc, s_rel, arcs, rels, mask, mbr=True, partial=False):
        arc_loss = SemanticsDist(s_arc, mask.sum(-1), arcs, 'dep').max_margin()
        # -1 denotes un-annotated arcs
        if partial:
            mask = mask & arcs.ge(0)
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(rels)), arcs[mask]]
        rel_loss = self.criterion(s_rel, rels)
        loss = arc_loss + rel_loss
        return loss


class MaxMarginConstModel(CRFConstituencyModel):
    def loss(self, s_span, s_label, charts, mask, mbr=True):
        span_mask = charts.ge(0) & mask
        span_loss = SemanticsDist(s_span, mask[:, 0].sum(-1), span_mask, 'con').max_margin()
        label_loss = self.criterion(s_label[span_mask], charts[span_mask])
        loss = span_loss + label_loss
        return loss

    # def decode(self, s_span, s_label, mask):
    #     r"""
    #     Args:
    #         s_span (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
    #             Scores of all constituents.
    #         s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
    #             Scores of all constituent labels.
    #         mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
    #             The mask for covering the unpadded tokens in each chart.

    #     Returns:
    #         List[List[Tuple]]:
    #             Sequences of factorized labeled trees.
    #     """

    #     # span_preds = ConstituencyCRF(s_span, mask[:, 0].sum(-1)).argmax
    #     # label_preds = s_label.argmax(-1).tolist()
    #     batch_size, seq_len, _ = s_span.shape
    #     lex_tree = BiLexicalizedDist(
    #         [s_span.new_zeros((batch_size, seq_len, seq_len)), s_span,
    #          s_span.new_zeros((batch_size, seq_len, seq_len, seq_len)),
    #          s_span.new_zeros((batch_size, seq_len, seq_len, seq_len))],
    #         mask[:, 0].sum(-1)).argmax
    #     label_preds = label_preds = s_label.argmax(-1).tolist()
    #     return [[(i, j, labels[i][j]) for i, j in spans] for spans, labels in zip(lex_tree[1], label_preds)]
