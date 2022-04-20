import spade.models as models
import streamlit as st
import json
from google.cloud import vision
import os
from transformers import AutoConfig
import spade.transforms as transforms
import cProfile
os.system("clear")

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] =\
    '/home/hung/grooo-gkeys.json'

st.header("Trích xuất hóa đơn")
st.warning("Hệ thống chưa được hoàn thiện, hiện tại mới chỉ có chức năng OCR, chưa có trích xuất thông tin")


def ocr(content: bytes):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    return json.loads(type(response).to_json(response))


@st.experimental_singleton
def get_model():
    config = AutoConfig.from_pretrained("vinai/phobert-base")
    model = models.BrosSpade(config, fields=[
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
        "total.change_label"])
    return model


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

if image is not None:
    st.image(image)
    submit = st.button("Nhận dạng")
    clear = st.button("clear")
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
#         cProfile.run("""
# models.preprocess({
#     "bbox_type": "xy4",
#     "tokenizer": "vinai/phobert-base",
#     "max_position_embeddings": 258
# }, input_data)
#         """)

        for (k, v) in batch.items():
            print(k, v.shape)

    with st.spinner("Initializing model..."):
        model = get_model()

    with st.spinner("Inferring..."):
        output = model(batch)
        print("Output", output)

    # with open("response.json", 'w') as f:
    #     f.write(json.dumps(res, ensure_ascii=False, indent=2))
    # st.write(json.dumps(res, ensure_ascii=False, indent=2))
