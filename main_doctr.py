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
from detect.ocr import *
import spade.transforms as transforms
from spade.models import SpadeData
from spade.bros.bros import BrosConfig
import time
from spade.output_convert import *
# checkpoint_path="../../spade-rewrite/checkpoint-bros-vnbill/best_score_parse_vnbill.pt"
checkpoint_path="../../spade-rewrite/weight/best_score_parse_invoice.pt"
os.system("clear")
st.set_page_config(layout="wide")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] =\
    '../../Invoice_template/key/grooo-gkeys.json'

st.header("Trích xuất hóa đơn")
# config = Cfg.load_config_from_name('vgg_transformer')
config = Cfg.load_config_from_name('vgg_seq2seq')
# config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['weights'] = 'https://drive.google.com/uc?id=1nTKlEog9YFK74kPyX0qLwCWi60_YHHk4'
config['cnn']['pretrained']=False
config['device'] = 'cuda:0'
config['predictor']['beamsearch']=False
detector_vietocr = Predictor(config)


@st.cache
def ocr(content: bytes):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    return json.loads(type(response).to_json(response))

def ocr_doctr(image):
    model_doctr = detection_predictor(arch='db_resnet50', pretrained=True,assume_straight_pages=True)
    doct_img=DocumentFile.from_images(image)
    a=time.time()
    result=model_doctr(doct_img)
    b=time.time()
    st.write(f"Doctr: {b-a}")
    img_copy=doct_img[0].copy()
    h,w,c=doct_img[0].shape
    bboxes=[]
    for box in result[0]:
        x1=int(box[0]*w)
        y1=int(box[1]*h)
        x2=int(box[2]*w)
        y2=int(box[3]*h)
        # bboxes.append([x1,x2,y1,y2])
        bboxes.insert(0,[x1,x2,y1,y2])
        img_copy=bounding_box(x1,y1,x2,y2,img_copy)
    st.image(img_copy, caption='Boxed_image')
    a=time.time()
    raw_text=Vietocr_img(img_copy,bboxes,detector_vietocr)
    b=time.time()
    st.write(f"Vietocr: {b-a}")
    
    g=arrange_bbox(bboxes)
    rows = arrange_row(g= g)

    for row in rows:
        x=""
        for i in row:
            x=x+" "+raw_text[i]
        st.write(x)

    return bboxes,raw_text,h,w

fields = [
    "info.date",
    "info.form",
    "info.serial",
    "info.num",
    "info.sign_date"
    "seller.name",
    "seller.company",
    "seller.tax",
    "seller.tel",
    "seller.address",
    "seller.bank",
    "customer.name",
    "customer.company",
    "customer.tax",
    "customer.tel",
    "customer.address",
    "customer.bank",
    "customer.payment_method",
    "menu.id",
    "menu.description",
    "menu.unit",
    "menu.quantity",
    "menu.unit_price",
    "menu.subtotal",
    "menu.vat_rate",
    "menu.vat",
    "menu.total",
    "total.subtotal",
    "total.vat_rate",
    "total.vat",
    "total.total"]

# fields = [
#     "store.name",
#     "store.address",
#     "store.phone",
#     "menu.name",
#     "menu.id",
#     "menu.count",
#     "menu.unit",
#     "menu.unitprice",
#     "menu.price",
#     "menu.discount",
#     "subtotal.tax",
#     "subtotal.count",
#     "subtotal.discount",
#     "subtotal.service",
#     "subtotal.price",
#     "total.price",
#     "total.currency",
#     "total.cash",
#     "total.credit",
#     "total.change",
#     "info.transaction",
#     "info.customer",
#     "info.time",
#     "info.staff",
#     "total.price_label",
#     "total.cash_label",
#     "total.change_label"]


@st.experimental_singleton
def get_model():
    tokenizer = AutoConfig.from_pretrained("vinai/phobert-base")

    NUM_HIDDEN_LAYERS=9
    MAX_POSITION_EMBEDDINGS = 700
    if NUM_HIDDEN_LAYERS > 0:
        config_bert = BrosConfig.from_pretrained(
            "naver-clova-ocr/bros-base-uncased",
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            vocab_size=tokenizer.vocab_size-1,
        )
    else:
        config_bert = BrosConfig.from_pretrained(
            "naver-clova-ocr/bros-base-uncased",
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            vocab_size=tokenizer.vocab_size-1,
        )
    model = models.BrosSpade(config_bert, fields=fields)
    sd = torch.load(checkpoint_path, map_location='cpu')
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
        bboxes,raw_text,h,w=ocr_doctr(image.getvalue())
        input_data=transforms.from_doctr(bboxes,raw_text,h,w)
        # res = ocr(image.getvalue())
        # input_data = transforms.from_google(res)
        # c = time.time()
        # print("GG-API: ",c-b)

    a=time.time()
    with st.spinner(text="Extracting features..."):
        
        batch = models.preprocess({
            "bbox_type": "xy4",
            "tokenizer": "vinai/phobert-base",
            "max_position_embeddings": 700
        }, input_data)

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

        data_new=data_convert(final_output)
        all_data={}
        main_info=get_main_info(data_new)
        menu_true,menu_false=get_menu(data_new)
        total,total_sign=get_total(data_new)
        all_data["main_info"]=main_info
        all_data["menu_true"]=menu_true
        all_data["menu_false"]=menu_false
        all_data["total"]=total

        with open(f"./temp_output.json","w") as f:
       
            json.dump(all_data,f,indent=2, ensure_ascii=False)
        right.code(json.dumps(final_output, ensure_ascii=False, indent=2))
    b=time.time()
    st.write(f"Spade: {b-a}")
    


    with right:
        # with st.container():
        with open("./temp_output.json","r") as f:
            data_new= json.load(f)


        all_key_list=list(data_new.keys())
        new_info={}

        for key in all_key_list:
            sub_key_list=list(data_new[key].keys())
            new_info[key]={}
            for sub_key in sub_key_list:
                new_info[key][sub_key] = st.text_input(f"{key} - {sub_key}",data_new[key][sub_key])
        
        button = st.button("Add")
        if button:
            for key in all_key_list:
                data_new[key]=new_info[key]
            with open(f"./temp_output.json","w") as f:
                json.dump(data_new,f,indent=2, ensure_ascii=False)
        st.write(data_new)
            