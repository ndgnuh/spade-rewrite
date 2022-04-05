import torch
from torch import nn, Tensor


class SinCosPositionalEncoding(nn.Module):
    def __init__(self, dim):
        # ws: Same as Transformer inv_freq = 1 / 10000^(-j / dim) . why log?
        super().__init__()

        self.dim = dim
        inv_freq = 1 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        shape = x.shape  # [B, T]
        x = x.view(-1).float()
        y = self.inv_freq
        sinusoid_in = torch.ger(x, y)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb.detach()


def flatten_tril(x):
    """
    Return the n * (n - 1) / 2 vectors of upper triangle
    """
    b, n, _ = x.shape
    r_idx, c_idx = torch.tril_indices(n, n)
    idx = r_idx * n + c_idx
    return x.contiguous().view(b, -1)[:, idx]


class ExperimentalRelationTagger(nn.Module):
    def __init__(self, n_fields, hidden_size, head_p_dropout=0.1):
        super().__init__()
        self.head = nn.Linear(hidden_size, hidden_size)
        self.tail = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.field_embeddings = nn.Parameter(torch.rand(1, n_fields, hidden_size))
        self.W_label_0 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_label_1 = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, enc, rel_embeds):
        enc_head = self.head(enc)
        enc_tail = self.tail(enc)
        value = self.value(rel_embeds)

        batch_size = enc_tail.size(0)
        field_embeddings = self.field_embeddings.expand(batch_size, -1, -1)
        enc_head = torch.cat([field_embeddings, enc_head], dim=1)
        value = torch.cat([field_embeddings, value], dim=1)

        attention_0 = torch.matmul(enc_head, self.W_label_0(enc_tail).transpose(1, 2))
        attention_1 = torch.matmul(enc_head, self.W_label_1(enc_tail).transpose(1, 2))

        score_0 = torch.einsum("bnm,blh->blm", attention_0, value)
        score_1 = torch.einsum("bnm,blh->blm", attention_1, value)

        score = torch.cat([score_0.unsqueeze(1), score_1.unsqueeze(1)], dim=1)
        return score


class RelativePositionEmbeddings(nn.Module):
    def __init__(self, sequence, hidden_size):
        super().__init__()
        assert hidden_size % 4 == 0
        self.x = nn.Linear(sequence, hidden_size // 4)
        self.y = nn.Linear(sequence, hidden_size // 4)
        self.distances = nn.Linear(sequence, hidden_size // 4)
        self.angles = nn.Linear(sequence, hidden_size // 4)

    def forward(self, x, y, distances, angles):
        x_embeds = self.x(x)
        y_embeds = self.x(y)
        distances_embeds = self.x(distances)
        angles_embeds = self.x(angles)
        return torch.cat([x_embeds, y_embeds, distances_embeds, angles_embeds], dim=-1)


def relative_vectors(bbox, normalization_term=1000):
    """
    Return relative vectors from (nbox * 4 * 2) bbox tensor. Bbox is [x1, y1, x2, y2]
    """
    # direction vector
    tl = bbox[:, 0]
    tr = bbox[:, 1]
    br = bbox[:, 2]
    bl = bbox[:, 3]
    directions = (tr + br) / 2 - (tl + bl) / 2
    direction_norms = directions.norm(dim=-1)

    # Relative angles
    angles = torch.einsum("ni,mi->nm", directions, directions)
    print(angles.shape)
    angles = angles / direction_norms.unsqueeze(0) / direction_norms.unsqueeze(1)
    angles = torch.clamp(angles, -1, 1)  # truncation error
    angles = angles.arccos()

    # Relative centers
    true_centers = bbox.mean(dim=1)
    centers = true_centers.unsqueeze(1) - true_centers.unsqueeze(0)
    x = centers[:, :, 0]
    y = centers[:, :, 1]

    # Relative distance
    distances = centers.norm(dim=-1)
    result = dict(
        x=x.unsqueeze(0) / normalization_term,
        y=y.unsqueeze(0) / normalization_term,
        distances=distances.unsqueeze(0) / normalization_term,
        angles=angles.unsqueeze(0),
    )
    return result


class SpadeEmbeddings(nn.Module):
    def __init__(self, config):
        self.components = getattr(
            config, "components", ["sequence_position", "relative_position"]
        )
        self.distance_unit = getattr(config, "distance_unit", 128)
        self.character_size_unit = getattr(config, "character_size_unit", 128)
        self.config = config

        num_positions = self.distance_unit * (self.distance_unit - 1) / 2

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.x_position_embeddings = nn.Embedding(num_position, config.hidden_size)
        self.y_position_embeddings = nn.Embedding(num_position, config.hidden_size)
        self.center_position_embeddings = nn.Embedding(num_position, config.hidden_size)
        self.distance_position_embeddings = nn.Embedding(
            num_position, config.hidden_size
        )
        self.angle_position_embeddings = nn.Embedding(num_position, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

    def forward(self, input_ids=None, bbox=None, token_type_ids=None):
        pass
