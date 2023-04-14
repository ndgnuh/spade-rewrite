import yaml

def validate_model_config(config):
    mandatory = ["type", "replace_word_embeddings"]
    for key in mandatory:
        assert key in config, f"Missing key {key} in model config"

def _read_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def read_model_config(config_file):
    config = read_config(config_file)
    validate_model_config(config)
    return config
