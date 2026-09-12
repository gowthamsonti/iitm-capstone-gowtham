import streamlit as st
import requests

st.title("Capstone project")
question = st.text_input("Ask Anything")

if question:
    placeholder = st.empty()
    answer = ""
    with requests.post("http://localhost:8000/ask",json={"question":question},stream=True) as r:
        for chunk in r.iter_content(decode_unicode=True):
            answer +=chunk
            placeholder.markdown(answer)