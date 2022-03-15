import torch
from torch import nn, Tensor


class SinCosPositionalEncoding(nn.Module):
    def __init__(self, dim):
        # ws: Same as Transformer inv_freq = 1 / 10000^(-j / dim) . why log?
        super().__init__()

        self.dim = dim
        inv_freq = 1 / \
            (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self,  x):
        shape = x.shape  # [B, T]
        x = x.view(-1).float()
        y = self.inv_freq
        sinusoid_in = torch.ger(x, y)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb.detach()


class SpatialTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.pad_token_id = config.pad_token_id

    def forward(self,
                text_tok_ids: Tensor,
                x: Tensor,
                y: Tensor,
                dist: Tensor,
                angle: Tensor,
                attention_mask: Tensor = None,
                token_type_ids: Tensor = None):
        if attention_mask is None:
            attention_mask = torch.ones_like(text_tok_ids, dtype=torch.float32)

        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1 - attention_mask) * -10000

        embeds, r_embeds = self.embeddings(
            text_tok_ids, x, y, dist, angle, attention_mask, token_type_ids)
        out = self.encoder(embeds, r_embeds, attention_mask=attention_mask)
        return out
        # return self.encoder(embeddings, pos_embedding, attention_mask)
