import torch
from supar.modules.pretrained import TransformerEmbedding


class TransformerPOSEmbedding(TransformerEmbedding):
    def __init__(self, *args, **kargs) -> TransformerEmbedding:
        super().__init__(*args, **kargs)
        self.POS_emb = None

    def set_POS(self, POS_emb):
        self.POS_emb = POS_emb

    def forward(self, tokens: torch.Tensor, feats=None) -> torch.Tensor:
        emb_trans = super().forward(tokens)
        if self.POS_emb is not None and feats is not None:
            emb_trans += self.POS_emb(feats[0])
        return emb_trans
