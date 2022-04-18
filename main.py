import spade.models as models
import streamlit as st

st.header("Trích xuất hóa đơn")

with st.spinner(text="Loading tokenizer"):
    tokenizer = st.cache(models.AutoTokenizer)(
        "vinai/phobert-base", local_files_only=True)
    st.success("Tokenizer loaded")

upload_methods = ["Từ thư viện trong máy", "Chụp ảnh mới"]
upload_method = st.selectbox("Phương pháp upload ảnh", upload_methods)


if upload_methods.index(upload_method) == 0:
    image = st.file_uploader("Upload file")
else:
    image = st.camera_input("Chụp ảnh")
print(upload_method)

c1, c2 = st.columns((1, 1))
with c1:
    submit = st.button("Nhận dạng")
clear = c2.button("clear")
