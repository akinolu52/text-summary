import pickle
import re
from io import BytesIO

import PyPDF2
import numpy as np
import requests
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model and components
model = load_model('./models2/sequence_2_sequence.keras')
encoder_model = load_model('./models2/encoder_model.keras')
decoder_model = load_model('./models2/decoder_model.keras')

with open('./models2/model_components.pkl', 'rb') as f:
    components = pickle.load(f)

x_tokenizer = components['x_tokenizer']
reverse_target_word_index = components['reverse_target_word_index']
target_word_index = components['target_word_index']
max_text_len = components['max_text_len']
max_summary_len = components['max_summary_len']


# Function to generate summary
def generate_summary(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if sampled_token != 'eostok':
            decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find stop word.
        if sampled_token == 'eostok' or len(decoded_sentence.split()) >= (max_summary_len - 1):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


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

        # Preprocess the input
        input_seq = x_tokenizer.texts_to_sequences([text_to_process])
        input_seq = pad_sequences(input_seq, maxlen=max_text_len, padding='post')

        # Generate summary
        abstractive_summary = generate_summary(input_seq)

        st.subheader("Abstractive Text")
        st.write(abstractive_summary)
    else:
        st.error("Please enter some text before processing.")

