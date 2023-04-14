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


class SpadeInputEmbeddings(nn.Module):
    """Based on BertEmbeddings of hugginface's"""

    def __init__(self, cfg, n_dist_unit=120, n_char_unit=120, input_embedding_components=None):
        super().__init__()
        if input_embedding_components is None:
            input_embedding_components = ["seqPos", "absPos"]

        self.word_embeddings = nn.Embedding(
            cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id
        )
        self.token_type_embeddings = nn.Embedding(
            cfg.type_vocab_size, cfg.hidden_size)

        # 2D embedding in input
        assert isinstance(input_embedding_components, list)
        self.input_embedding_components = input_embedding_components
        if "seqPos" in self.input_embedding_components:
            self.position_embeddings = nn.Embedding(
                cfg.max_position_embeddings, cfg.hidden_size
            )

            # position_ids (1, len position emb) is contiguous in memory and exported when serialized
            self.register_buffer(
                "position_ids",
                torch.arange(cfg.max_position_embeddings).expand((1, -1)),
            )
            self.position_embedding_type = getattr(
                cfg, "position_embedding_type", "absolute"
            )

        self.n_dist_unit = n_dist_unit
        n_pos = n_dist_unit * 2 + 1

        if "absPos" in self.input_embedding_components:
            print("abs position added in the input")
            self.pos_x_embeddings = nn.Embedding(
                n_pos, cfg.hidden_size, _weight=torch.zeros(
                    n_pos, cfg.hidden_size)
            )
            self.pos_y_embeddings = nn.Embedding(
                n_pos, cfg.hidden_size, _weight=torch.zeros(
                    n_pos, cfg.hidden_size)
            )
        # if "charSize" in self.input_embedding_components:
        #     self.char_size_embeddings = nn.Embedding(
        #         n_char_unit,
        #         cfg.hidden_size,
        #         _weight=torch.zeros(n_char_unit, cfg.hidden_size),
        #     )
        # if "vertical" in self.input_embedding_components:
        #     self.vertical_embeddings = nn.Embedding(
        #         2, cfg.hidden_size, _weight=torch.zeros(2, cfg.hidden_size)
        #     )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        position_ids,
        token_type_ids=None,
        **kwargs
    ):

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        rn_center_x_ids, rn_center_y_ids = position_ids

        words_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + token_type_embeddings
        if "seqPos" in self.input_embedding_components:
            seq_length = input_ids.size(1)
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        if "absPos" in self.input_embedding_components:
            pos_x_embeddings = self.pos_x_embeddings(
                rn_center_x_ids[:, 0] + self.n_dist_unit
            )  # use first token as an origin
            pos_y_embeddings = self.pos_y_embeddings(
                rn_center_y_ids[:, 0] + self.n_dist_unit
            )

            embeddings += pos_x_embeddings
            embeddings += pos_y_embeddings

        # if "charSize" in self.input_embedding_components:
        #     char_size_embeddings = self.char_size_embeddings(vertical_ids)
        #     embeddings += char_size_embeddings
            # print('cc')
        # if "vertical" in self.input_embedding_components:
        #     vertical_embeddings = self.vertical_embeddings(vertical_ids)
        #     embeddings += vertical_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


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
