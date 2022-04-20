import torch
import spade.models as models
import streamlit as st
import json
from google.cloud import vision
import os
from transformers import AutoConfig
import spade.transforms as transforms
import cProfile
from pprint import pformat
os.system("clear")
st.set_page_config(layout="wide")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] =\
    '/home/hung/grooo-gkeys.json'

st.header("Trích xuất hóa đơn")


@st.cache
def ocr(content: bytes):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    return json.loads(type(response).to_json(response))


fields = [
    "store.name",
    "store.address",
    "store.phone",
    "menu.name",
    "menu.id",
    "menu.count",
    "menu.unit",
    "menu.unitprice",
    "menu.price",
    "menu.discount",
    "subtotal.tax",
    "subtotal.count",
    "subtotal.discount",
    "subtotal.service",
    "subtotal.price",
    "total.price",
    "total.currency",
    "total.cash",
    "total.credit",
    "total.change",
    "info.transaction",
    "info.customer",
    "info.time",
    "info.staff",
    "total.price_label",
    "total.cash_label",
    "total.change_label"]


@st.experimental_singleton
def get_model():
    config = AutoConfig.from_pretrained("vinai/phobert-base")
    model = models.BrosSpade(config, fields=fields)
    sd = torch.load("best_score_parse.pt", map_location='cpu')
    model.load_state_dict(sd, strict=False)
    return model


with st.spinner(text="Loading model"):
    model = get_model()
    st.success("Model loaded")

with st.spinner(text="Loading tokenizer"):
    tokenizer = st.experimental_singleton(models.AutoTokenizer)(
        "vinai/phobert-base", local_files_only=True)
    st.success("Tokenizer loaded")

upload_methods = ["Từ thư viện trong máy", "Chụp ảnh mới"]
upload_method = st.radio("Phương pháp upload ảnh", upload_methods)


if upload_methods.index(upload_method) == 0:
    image = st.file_uploader("Upload file")
else:
    image = st.camera_input("Chụp ảnh")

left, right = st.columns(2)
if image is not None:
    left.image(image)
    submit = left.button("Nhận dạng")
    clear = left.button("clear")
else:
    submit = clear = False

if submit:
    with st.spinner(text="OCR..."):
        res = ocr(image.getvalue())
        input_data = transforms.from_google(res)

    with st.spinner(text="Extracting features..."):
        import time
        a = time.time()
        batch = models.preprocess({
            "bbox_type": "xy4",
            "tokenizer": "vinai/phobert-base",
            "max_position_embeddings": 258
        }, input_data)
        b = time.time()
        print("Time", (b - a))

        for (k, v) in batch.items():
            print(k, v.shape)

    with st.spinner("Inferring..."):
        output = model(batch)

    with st.spinner("Post processing..."):
        final_output = models.post_process(
            tokenizer,
            relations=output.relations,
            batch=batch,
            fields=fields
        )
        right.code(json.dumps(final_output, ensure_ascii=False, indent=2))
