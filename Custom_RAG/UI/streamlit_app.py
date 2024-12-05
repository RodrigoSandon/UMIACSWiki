import streamlit as st
import requests

st.title("UMIACS Wiki Chatbot")

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question:
        response = requests.post(
            "http://192.168.14.31:5000/chat",  # Update this to the Flask server's IP
            json={"question": question}
        )

        if response.status_code == 200:
            st.write("Answer:", response.json().get("answer"))
        else:
            st.error("Error: " + response.text)
