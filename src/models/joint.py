import torch
import torch.nn as nn
# from supar.model import Model
from supar.modules import Biaffine, Triaffine
from supar.structs import (DependencyCRF, MatrixTree, ConstituencyCRF)
from supar.utils import Config
from supar.utils.common import MIN
from utils.modules import Span2oScorer, BiLexicalizedDist, LexMFVI
from utils.modules import SemanticsDist
from .model import Model


class JointModel(Model):

    def __init__(self,
                 n_words,
                 n_rels,
                 n_labels,
                 n_tags=None,
                 n_chars=None,
                 encoder='lstm',
                 feat=['char'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 elmo='original_5b',
                 elmo_bos_eos=(True, False),
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 finetune=False,
                 n_plm_embed=0,
                 embed_dropout=.33,
                 n_encoder_hidden=800,
                 n_encoder_layers=3,
                 encoder_dropout=.33,
                 n_arc_mlp=500,
                 n_rel_mlp=100,
                 n_span_mlp=500,
                 n_label_mlp=100,
                 n_dsp_mlp=500,
                 n_sib_mlp=100,
                 n_pair_mlp=100,
                 n_head_mlp=100,
                 mlp_dropout=.33,
                 scale=0,

                 mfvi=False,
                 max_iter=3,
                 use_head=False,
                 use_dsp=False,
                 use_sib=False,
                 use_pair=False,
                 loss_type='nl',
                 dsp_scorer='biaffine',
                 head_scorer='biaffine',
                 dual_dsp=True,
                 structured=False,

                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))
        self.arc_attn = Biaffine(n_in=self.args.n_encoder_hidden, n_proj=self.args.n_arc_mlp,
                                 scale=scale, bias_x=True, bias_y=False, dropout=self.args.mlp_dropout)
        self.rel_attn = Biaffine(n_in=self.args.n_encoder_hidden, n_proj=self.args.n_rel_mlp,
                                 n_out=n_rels, bias_x=True, bias_y=True, dropout=self.args.mlp_dropout)
        self.span_attn = Biaffine(n_in=self.args.n_encoder_hidden, n_proj=self.args.n_span_mlp,
                                  bias_x=True, bias_y=False, dropout=self.args.mlp_dropout)
        self.label_attn = Biaffine(n_in=self.args.n_encoder_hidden, n_proj=self.args.n_label_mlp, n_out=n_labels,
                                   bias_x=True, bias_y=True, dropout=self.args.mlp_dropout)

        if self.args.use_head:
            self.head_attn = Span2oScorer(n_in=self.args.n_encoder_hidden, n_mlp=self.args.n_head_mlp,
                                          scale=self.args.scale, dropout=self.args.mlp_dropout,
                                          score_type=self.args.head_scorer, mask='head')
            self.hook_attn = Span2oScorer(n_in=self.args.n_encoder_hidden, n_mlp=self.args.n_head_mlp,
                                          scale=self.args.scale, dropout=self.args.mlp_dropout,
                                          score_type=self.args.head_scorer, mask='hook')

        # mfvi
        if self.args.mfvi:
            # high-order scores
            if self.args.use_dsp:
                if self.args.dual_dsp:
                    self.dsp_scorer_l = Span2oScorer(n_in=self.args.n_encoder_hidden, n_mlp=self.args.n_dsp_mlp,
                                                     scale=self.args.scale, dropout=self.args.mlp_dropout, score_type=self.args.dsp_scorer)
                    self.dsp_scorer_r = Span2oScorer(n_in=self.args.n_encoder_hidden, n_mlp=self.args.n_dsp_mlp,
                                                     scale=self.args.scale, dropout=self.args.mlp_dropout, score_type=self.args.dsp_scorer)
                else:
                    self.dsp_scorer = Span2oScorer(n_in=self.args.n_encoder_hidden, n_mlp=self.args.n_dsp_mlp,
                                                   scale=self.args.scale, dropout=self.args.mlp_dropout, score_type=self.args.dsp_scorer)
            if self.args.use_sib:
                self.sib_attn = Triaffine(n_in=self.args.n_encoder_hidden, n_proj=self.args.n_sib_mlp,
                                          scale=self.args.scale, bias_x=True, bias_y=True, dropout=self.args.mlp_dropout)
            if self.args.use_pair:
                self.pair_attn = Triaffine(n_in=self.args.n_encoder_hidden, n_proj=self.args.n_pair_mlp,
                                           bias_x=True, bias_y=True, dropout=self.args.mlp_dropout)
            self.inf = LexMFVI(self.args.max_iter,
                               self.args.structured, self.args.loss_type)

        # label loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words, feats=None):
        x = self.encode(words, feats)

        # dependency forward
        mask = words.ne(self.args.pad_index) if len(words.shape) < 3 else words.ne(self.args.pad_index).any(-1)
        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(x, x).float().masked_fill_(
            ~mask.unsqueeze(1), MIN)[:, :-1, :-1]
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(x, x).permute(0, 2, 3, 1)[:, :-1, :-1, :]

        # constituency forward
        # [batch_size, seq_len, seq_len]
        x_f, x_b = x.chunk(2, -1)
        x_bdy = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)
        s_span = self.span_attn(x_bdy, x_bdy).float()
        # [batch_size, seq_len, seq_len, n_labels]
        s_label = self.label_attn(x_bdy, x_bdy).permute(0, 2, 3, 1)

        # high-order feature forward
        s_vi2o = {}
        if self.args.mfvi:
            if self.args.use_dsp:
                if self.args.dual_dsp:
                    s_dsp_l = self.dsp_scorer_l(x, x)
                    s_dsp_r = self.dsp_scorer_r(x, x)
                    s_dsp = (s_dsp_l, s_dsp_r)
                else:
                    s_dsp = self.dsp_scorer(x, x)
                s_vi2o['dsp'] = s_dsp
            if self.args.use_sib:
                # [batch_size, seq_len, seq_len, seq_len] view input as (s, m, h) and output (-, m, h, s)
                s_vi2o['sib'] = self.sib_attn(x, x, x).permute(0, 3, 1, 2)[
                    :, :-1, :-1, :-1]
            if self.args.use_pair:
                # [batch_size, seq_len, seq_len, seq_len] view input as (l, r, b) and output (-, l, r, b)
                s_vi2o['pair'] = self.pair_attn(
                    x_bdy, x_bdy, x_bdy).permute(0, 2, 3, 1)

        if self.args.use_head:
            s_head = self.head_attn(x, x_bdy)
            s_hook = self.hook_attn(x, x_bdy)
            return (s_arc, s_span, s_head, s_hook), (s_rel, s_label), s_vi2o

        return (s_arc, s_span), (s_rel, s_label), s_vi2o

    def loss(self, s_lex, s_tag, gold, s_vi2o, mask_dep, mask_con, mbr=True):
        if self.args.mfvi:
            return self.loss_mfvi(s_lex, s_tag, gold, s_vi2o, mask_dep, mask_con, mbr=mbr)
        else:
            if self.args.loss_type in ['nl', 'mm']:
                return self.loss_lex(s_lex, s_tag, gold, mask_dep, mask_con, partial=False, mbr=mbr)
            elif self.args.loss_type == 'hb':
                arcs, charts, rels = gold
                dep_loss = self.loss_dep(
                    s_lex[0], s_tag[0], arcs, rels, mask_dep, partial=False, mbr=mbr)
                con_loss = self.loss_con(
                    s_lex[1], s_tag[1], charts, mask_con, mbr=mbr)
                return dep_loss+con_loss, None
            else:
                raise ValueError(f"Unknown loss type: {self.args.loss_type}")

    def loss_mfvi(self, s_lex, s_tag, gold, s_vi2o, mask_dep, mask_con, mbr=True):
        s_rel, s_label = s_tag
        arcs, charts, rels = gold[:3]
        span_mask = charts.ge(0) & mask_con
        lex_gold = (arcs, span_mask, gold[3]) if self.args.use_head else (
            arcs, span_mask)
        lex_loss, marginals = self.inf(
            s_lex, s_vi2o, (mask_dep, mask_con), lex_gold)
        marginals = marginals if mbr else s_lex

        s_rel, rels = s_rel[mask_dep], rels[mask_dep]
        s_rel = s_rel[torch.arange(len(rels)), arcs[mask_dep]]
        rel_loss = self.criterion(s_rel, rels)
        label_loss = self.criterion(s_label[span_mask], charts[span_mask])

        loss = rel_loss + label_loss + lex_loss

        return loss, marginals

    def loss_lex(self, s_lex, s_tag, gold, mask_dep, mask_con, partial=False, mbr=True, lambda_=0.5):
        r"""The loss of lexical structures which contain crf neg likelihood and max margin"""
        s_rel, s_label = s_tag
        arcs, charts, rels = gold[:3]
        span_mask = charts.ge(0) & mask_con
        lex_gold = (arcs, span_mask, *
                    gold[3:]) if self.args.use_head else (arcs, span_mask)
        lex_dist = BiLexicalizedDist(s_lex, mask_dep.sum(1), lex_gold)

        if self.args.loss_type == 'nl':
            lex_loss = lex_dist.neg_likelihoods()
            marginals = lex_dist.marginals if mbr else s_lex
        else:
            lex_loss = lex_dist.max_margin()
            marginals = s_lex
        if partial:
            mask_dep = mask_dep & arcs.ge(0)
        s_rel, rels = s_rel[mask_dep], rels[mask_dep]
        s_rel = s_rel[torch.arange(len(rels)), arcs[mask_dep]]
        rel_loss = self.criterion(s_rel, rels)
        label_loss = self.criterion(s_label[span_mask], charts[span_mask])
        loss = lex_loss * lambda_ + (rel_loss+label_loss) * (1. - lambda_)
        return loss, marginals

    def loss_dep(self, s_arc, s_rel, arcs, rels, mask, partial=False, mbr=True):
        arc_loss = SemanticsDist(s_arc, mask.sum(-1), arcs, 'dep').max_margin()
        # -1 denotes un-annotated arcs
        if partial:
            mask = mask & arcs.ge(0)
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(rels)), arcs[mask]]
        rel_loss = self.criterion(s_rel, rels)
        loss = arc_loss + rel_loss
        return loss

    def loss_con(self, s_span, s_label, charts, mask, mbr=True):
        span_mask = charts.ge(0) & mask
        span_loss = SemanticsDist(
            s_span, mask[:, 0].sum(-1), span_mask, 'con').max_margin()
        label_loss = self.criterion(s_label[span_mask], charts[span_mask])
        loss = span_loss + label_loss
        return loss

    def decode(self, s_lex, s_tag, mask_dep, mask_con, decoding='satta', mbr=True):
        if decoding == 'satta':
            # print([x.shape for x in s_lex])
            lex_tree = BiLexicalizedDist(s_lex, mask_dep.sum(1)).argmax
            rel_preds = s_tag[0].argmax(-1).gather(-1,
                                                   lex_tree[0].unsqueeze(-1)).squeeze(-1)
            label_preds = s_tag[1].argmax(-1).tolist()
            arc_preds = lex_tree[0]
            chart_preds = [[(i, j, labels[i][j]) for i, j in spans]
                           for spans, labels in zip(lex_tree[1], label_preds)]
        elif decoding == 'idp':
            arc_preds, rel_preds = self.decode_dep(
                s_lex[0], s_tag[0], mask_dep, tree=True, proj=True)
            chart_preds = self.decode_con(s_lex[1], s_tag[1], mask_con)
        else:
            raise KeyError(f"Unknown decoding type {decoding}. ")

        return arc_preds, rel_preds, chart_preds

    def decode_dep(self, s_arc, s_rel, mask, tree=False, proj=False):
        # lens = mask.sum(1)
        # arc_preds = s_arc.argmax(-1)
        # bad = [not CoNLL.istree(seq[1:i+1], proj)
        #        for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        # if tree and any(bad):
        arc_preds = (DependencyCRF if proj else MatrixTree)(s_arc, mask.sum(-1)).argmax
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds

    def decode_con(self, s_span, s_label, mask):
        span_preds = ConstituencyCRF(s_span, mask[:, 0].sum(-1)).argmax
        label_preds = s_label.argmax(-1).tolist()
        return [[(i, j, labels[i][j]) for i, j in spans] for spans, labels in zip(span_preds, label_preds)]
