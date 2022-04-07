from traceback import print_exc
import spade.utils as utils
from fastapi import FastAPI, Request
from spade.model_layoutlm import SpadeDataset, LayoutLMSpade, post_process
from spade.utils import AttrDict
import uvicorn
import json
from transformers import AutoTokenizer, AutoModel, AutoConfig
from argparse import Namespace, ArgumentParser
import torch
from torch import mode


def predict_json(context, json_: str):
    """
    Return JSON prediction Result from JSON string from body
    """
    dataset = SpadeDataset(
        context.tokenizer, context.backbone_config, json_.splitlines())
    for b in range(len(dataset)):
        batch = dataset[b]
        with torch.no_grad():
            raw_output = context.model(batch)

            final_output = post_process(
                tokenizer=context.tokenizer,
                rel_s=raw_output.relations[0][b],
                rel_g=raw_output.relations[1][b],
                fields=dataset.fields,
                batch=batch
            )

    return final_output


def create_app(config):
    app = FastAPI()
    context = Namespace()
    context.config = config
    context.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    context.backbone_config = AutoConfig.from_pretrained("vinai/phobert-base")

    # Model
    context.model = LayoutLMSpade(config)
    print(context.model)
    device = config.model.get("device", None)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = config.model.get("checkpoint", None)
    strict = config.model.get("checkpoint_strict", True)
    if checkpoint is not None:
        sd = torch.load(checkpoint, map_location=device)
        context.model.load_state_dict(sd, strict=strict)

    # Context name space for the app
    app.context = context

    @app.get("/configuration")
    def server_configuration():
        return context.config

    @app.post("/from-json")
    async def extract_json(req: Request):
        body = await req.body()
        content = body.decode('utf-8').strip()
        try:
            return predict_json(context, content)
        except Exception:
            print_exc()
            return 0
    return app


def start_app(app, *args, **kwargs):
    uvicorn.run(app, *args, **kwargs)


def main():
    config = utils.read_config("config/vi-invoice.yaml")

    app = create_app(config)
    start_app(app)


if __name__ == "__main__":
    main()
