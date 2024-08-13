import streamlit as st
import requests
import PyPDF2
from io import BytesIO
import re


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
    c = 0
    text = ""

    while c < len(pdf_reader.pages):
        pageObj = pdf_reader.pages[c]
        # text += pageObj.extract_text()
        page_text = pageObj.extract_text()
        if page_text:
            # Add line breaks based on common patterns
            page_text = re.sub(r'\n+', '\n', page_text)
            text += page_text + "\n"
        c += 1

    return text.replace('\n', ' ')


# Title of the app
st.title("Text Summarization App")

# Option to upload PDF or enter text
option = st.radio("Choose input method", ["Upload PDF", "Enter Text"])

# Handle PDF upload
if option == "Upload PDF":
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.text_area("Extracted Text", pdf_text, height=130, )
        text_to_process = pdf_text
    else:
        text_to_process = ""

# Handle direct text input
elif option == "Enter Text":
    text_to_process = st.text_area("Enter your paragraph to summarize here:", height=100)

# Button to process the text
if st.button("Summarize Text"):
    if text_to_process:
        response = requests.post(
            'http://127.0.0.1:5000/process',
            json={'paragraph': text_to_process}
        )

        if response.status_code == 200:
            result = response.json()
            st.subheader("Extractive Summary:")
            st.write(result.get('summary'))
        else:
            st.error(f"Error: {response.json().get('error')}")

        abstractive_text = "This is the abstractive summary based on the input text."

        st.subheader("Abstractive Text")
        st.write(abstractive_text)
    else:
        st.error("Please enter some text before processing.")

