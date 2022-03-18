import torch
from torch import nn
from transformers import AutoModel, AutoConfig


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

        # self.LayerNorm is not snake-cased
        # to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)


class GraphGenerator(nn.Module):
    def __init__(self, config, fields):
        super().__init__()
        self.n_fields = len(fields)
        self.fields = fields

        self.W_h = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_d = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_0 = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_1 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, score):
        h_part_1 = score[:, :self.n_fields, :]
        h_part_2 = self.W_h(score[:, self.n_fields:, :])
        h = torch.cat([h_part_1, h_part_2], dim=1)
        d = self.W_d(h_part_2)
        s1 = torch.einsum('bih,bjh->bij', h, self.W_0(d))
        s0 = torch.einsum('bih,bjh->bij', h, self.W_1(d))
        es0 = torch.exp(s0)
        es1 = torch.exp(s1)
        p = es0 / (es0 + es1)
        return p


class Spade(nn.Module):
    def __init__(self, bert_model, fields, n_dist_unit=1000):
        super().__init__()
        self.fields = fields
        self.cfg = cfg = AutoConfig.from_pretrained(bert_model)
        self.bert = AutoModel.from_pretrained(bert_model)
        bert_model.embeddings = SpadeInputEmbeddings(
            cfg, n_dist_unit=n_dist_unit)
        self.score_s = GraphGenerator(cfg, fields)
        self.score_g = GraphGenerator(cfg, fields)

    def forward(self, input_ids, position_ids, original_length, part_indices):
        encoded = torch.zeros(
            input_ids.shape[0], part_indices[-1].stop, self.cfg.hidden_size)

        # combine chunks
        for (i, slice_) in enumerate(part_indices):
            position_ids_ = {
                'x': position_ids['x'][:, i, :],
                'y': position_ids['y'][:, i, :]
            }
            input_ids_ = input_ids[:, i, :]
            encoded_i = self.bert(input_ids=input_ids_,
                                  position_ids=position_ids_).last_hidden_state
            # print(encoded_i.shape, encoded[:, slice_, :].shape, (slice_))
            encoded[:, slice_, :] += encoded_i

        # generate graph
        encoded = encoded[:, :original_length, :]
        score = torch.cat([
            self.score_s(encoded).unsqueeze(1),
            self.score_g(encoded).unsqueeze(1)
        ], dim=1)
        return score
