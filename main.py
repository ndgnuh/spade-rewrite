import math
from scipy import ndimage
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
from spade.bounding_box import Box
from functools import reduce
from PIL import Image

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/hung/grooo-gkeys.json'


def hough_lines_rotate(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0,
                            100, minLineLength=100, maxLineGap=5)
    angles = []

    for [[x1, y1, x2, y2]] in lines:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)
    print("Angle", median_angle)
    return ndimage.rotate(img, median_angle)


def exif_reorientation(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    # https://stackoverflow.com/questions/44537075/image-orientation-pythonopencv
    # this exif thing is black magic
    angle = {3: 180, 6: 270, 8: 90}.get(image.getexif().get(274, 0), 0)
    return image.rotate(angle)


def position_graph(bboxes):
    n = len(bboxes)
    centres = [b.center for b in bboxes]
    xcentres = [c[0] for c in centres]
    ycentres = [c[1] for c in centres]
    heights = [b.height for b in bboxes]
    width = [b.width for b in bboxes]

    def is_top_to(i, j):
        result = (ycentres[j] - ycentres[i]) > ((heights[i] + heights[j]) / 4)
        return result

    def is_left_to(i, j):
        return (xcentres[i] - xcentres[j]) > ((width[i] + width[j]) / 4)

    # <L-R><T-B>
    # +1: Left/Top
    # -1: Right/Bottom
    g = np.zeros((n, n), dtype='int')
    for i in range(n):
        for j in range(n):
            if is_left_to(i, j):
                g[i, j] += 10
            if is_left_to(j, i):
                g[i, j] -= 10
            if is_top_to(i, j):
                g[i, j] += 1
            if is_top_to(j, i):
                g[i, j] -= 1
    return g


def arrange_row(position_graph=None):
    n, m = position_graph.shape
    assert m == n
    visited = [False for i in range(n)]

    def row_indices(i: int):
        if visited[i]:
            return []
        visited[i] = True
        indices = [j for j in range(n)
                   if abs(position_graph[i, j]) == 10
                   and not visited[i]]
        indices = [i] + indices
        indices = np.array(indices)
        aux = position_graph[np.ix_(indices, indices)]
        order = np.argsort(np.sum(aux, axis=1))
        indices = indices[order].tolist()
        indices = [int(i) for i in indices]
        for i in indices:
            visited[i] = True
        return indices

    rows = []
    for i in range(n):
        if visited[i]:
            continue
        indices = row_indices(i)
        if len(indices) > 0:
            rows.append(indices)

    return rows


def ocr(content: bytes):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    return response


def convert2spadedata(response: vision.AnnotateFileResponse, width, height):
    j = type(response).to_json(response)
    with open("response.json", "w") as f:
        f.write(j)
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
    data['img_sz'] = {'width': width, 'height': height}

    # SORT BOUNDING BOXES INTO LEFT-RIGHT, UP/DOWN ORDER
    bboxes = [Box(b, Box.Type.QUAD) for b in data['coord']]
    sorted_indices = arrange_row(position_graph(bboxes))
    # for r in sorted_indices:
    #     print([data['text'][c] for c in r])
    sorted_indices = reduce(lambda x, y: x + y, sorted_indices, [])
    data['coord'] = [bboxes[i].xyxy for i in sorted_indices]
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
    # context.model = LayoutLMSpade(config)
    # print(context.model)
    # device = config.model.get("device", None)
    # if device is None:
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # checkpoint = config.model.get("checkpoint", None)
    # strict = config.model.get("checkpoint_strict", True)
    # if checkpoint is not None:
    #     sd = torch.load(checkpoint, map_location=device)
    #     context.model.load_state_dict(sd, strict=strict)

    # Context name space for the app
    app.context = context

    @ app.get("/configuration")
    def server_configuration():
        return context.config

    @ app.post("/from-image-no-rotate")
    async def _(file: UploadFile = File(...)):
        content = await file.read()
        img = np.fromstring(content, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        # OCR AND PREPARE DATA
        height, width, channel = img.shape
        data = convert2spadedata(ocr(content), width, height)
        with open("tmp.jsonl", "w") as f:
            f.write(json.dumps(data))
        return predict_json(context, [data])

    @ app.post("/from-image")
    async def _(file: UploadFile = File(...)):
        content = await file.read()
        img = np.fromstring(content, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = np.array(exif_reorientation(img))
        img = hough_lines_rotate(img)

        # OCR AND PREPARE DATA
        height, width, channel = img.shape
        data = convert2spadedata(ocr(content), width, height)
        with open("tmp.jsonl", "w") as f:
            f.write(json.dumps(data))
        return predict_json(context, [data])

    @ app.post("/from-json")
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
    start_app(app, host="0.0.0.0")


if __name__ == "__main__":
    main()
