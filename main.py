import spade.models as models
import streamlit as st
import json
from google.cloud import vision
import os
import spade.transforms as transforms

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] =\
    '/home/hung/grooo-gkeys.json'

st.header("Trích xuất hóa đơn")


def ocr(content: bytes):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    return json.loads(type(response).to_json(response))


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
        print(input_data)
    # with open("response.json", 'w') as f:
    #     f.write(json.dumps(res, ensure_ascii=False, indent=2))
    # st.write(json.dumps(res, ensure_ascii=False, indent=2))
