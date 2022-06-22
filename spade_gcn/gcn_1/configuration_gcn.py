from transformers.configuration_utils import PretrainedConfig


class ProtoGraphConfig(PretrainedConfig):

    def __init__(self,
                 n_layer=5,
                 d_model=768,
                 n_head=16,
                 vocab_size=64001,
                 n_labels=10,
                 p_dropout=0.1,
                 n_relation=2):
        super.__init__(n_layer=5,
                       d_model=768,
                       n_head=16,
                       vocab_size=64001,
                       n_labels=10,
                       p_dropout=0.1,
                       n_relation=2)
