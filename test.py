import streamlit as st
import numpy as np
@st.cache(allow_output_mutation=True)
def persistdata():
    return {}

d = persistdata()
d={"1":"trong",
"2":"ngoai"}
with st.container():
    st.write("This is inside the container")
    col1, col2 = st.columns(2)
    # You can call any Streamlit command, including custom components:
    st.bar_chart(np.random.randn(50, 3))
    with col1:
        k = st.text_input("Key","1")
    with col2:
        v = st.text_input("Value",d["1"])
    button = st.button("Add")
    if button:
        if k and v:
            d[k] = v
    st.write(d)


st.write("This is outside the container")



# with st.beta_container():
#     
#     col1, col2 = st.beta_columns(2)
#     with col1:
#         k = st.text_input("Key")
#     with col2:
#         v = st.text_input("Value")
#     button = st.button("Add")
#     if button:
#         if k and v:
#             d[k] = v
#     st.write(d)


# st.container()