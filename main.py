from traceback import print_exc
import spade.utils as utils
from fastapi import FastAPI, Request, UploadFile, File
from spade.model_layoutlm import SpadeDataset, LayoutLMSpade, post_process
from spade.utils import AttrDict
import uvicorn
import json
from transformers import AutoTokenizer, AutoModel, AutoConfig
from argparse import Namespace, ArgumentParser
import torch
from torch import mode
from google.cloud import vision
import os
import cv2
import numpy as np


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/hung/grooo-gkeys.json'


def ocr(content: bytes):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    return response


def convert2spadedata(response: vision.AnnotateFileResponse):
    j = type(response).to_json(response)
    j = json.loads(j)

    texts = j['textAnnotations']
    # Remove the overview box
    # TODO: add language support
    # texts[0]['locale']
    # print(texts[0])

    # TODO: cut the image to the size of the
    # Overview box and normalize the
    overview = texts.pop(0)

    data = {}
    data['text'] = []
    data['coord'] = []
    data['label'] = None
    for text in texts:
        data['text'].append(text['description'])
        data['coord'].append([[pt['x'], pt['y']]
                             for pt in text['boundingPoly']['vertices']])

    data['vertical'] = [False for _ in data['text']]
    data['img_feature'] = None
    data['img_url'] = None
    return data


def predict_json(context, jsonl: list):
    """
    Return JSON prediction Result from JSON string from body
    """
    dataset = SpadeDataset(context.tokenizer,
                           context.backbone_config,
                           jsonl,
                           fields=context.config.model.config.fields)
    for b in range(len(dataset)):
        batch = dataset[b]
        with torch.no_grad():
            raw_output = context.model(batch)

            final_output = post_process(
                tokenizer=context.tokenizer,
                rel_s=raw_output.relations[0][b],
                rel_g=raw_output.relations[1][b],
                fields=context.config.model.config.fields,
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

    @app.post("/from-image")
    async def _(file: UploadFile = File(...)):
        content = await file.read()
        img = np.fromstring(content, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        h, w, c = img.shape

        # OCR AND PREPARE DATA
        data = ocr(content)
        data = convert2spadedata(data)
        data['img_sz'] = {'width': w, 'height': h}
        return predict_json(context, [data])

    @app.post("/from-json")
    async def extract_json(req: Request):
        body = await req.body()
        content = body.decode('utf-8').strip()
        jsonl = [json.loads(line) for line in content.splitlines()]
        try:
            return predict_json(context, jsonl)
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
