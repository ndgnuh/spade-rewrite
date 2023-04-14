#!/usr/bin/env python3
from kie import configs
from argparse import ArgumentParser
import icecream
icecream.install()

def train(args):
    model_config = configs.read_model_config(args.model_config)
    config = configs.read_training_config(args.training_config)
    config.model_config = model_config

    from kie.models import Trainer
    trainer = Trainer(config)
    trainer.fit()

def main():
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers(title="action", dest="action", required=True)

    # Train arguments
    train_parser = sub_parsers.add_parser("train", help="Run train script")
    train_parser.add_argument(
        "--config", "-c",
        help="model config",
        dest="model_config",
        required=True)
    train_parser.add_argument(
        "--experiment", "-e",
        help="traininig config",
        dest="training_config",
        required=True)

    args = parser.parse_args()
    assert args.action in ["train"]

    if args.action == "train":
        train(args)

if __name__ == "__main__":
    main()
