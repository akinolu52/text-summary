from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import heapq
import re
from io import BytesIO

import PyPDF2
import nltk
import streamlit as st
import torch
from nltk.tokenize import sent_tokenize

nltk.download('punkt')


# Load the extractive model
@st.cache_resource
def load_extractive_model():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")


# load the abstractive model
@st.cache_resource
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    return model, tokenizer


extractive_model = load_extractive_model()

abstractive_model, abstractive_tokenizer = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
abstractive_model = abstractive_model.to(device)


def generate_abstractive_summary(text):
    max_summary_length = max(30, min(len(text.split()) // 2, 150))

    inputs = abstractive_tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(device)
    summary_ids = abstractive_model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=max_summary_length,
        min_length=30,
        length_penalty=2.0,
        early_stopping=True
    )
    summary = abstractive_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print(len(summary))
    return summary


def clean_text(text):
    text = text.strip()
    # remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    # remove \n and quotes like “, ‘, ”
    text = re.sub(r'[\n“”‘’]', '', text)

    return text


def compute_sentence_embeddings(sentences):
    embeddings = extractive_model.encode(sentences, convert_to_tensor=True)
    return embeddings


def generate_extractive_summary(text, summary_ratio=0.3):
    text = clean_text(text)

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        return ""

    # Compute sentence embeddings
    sentence_embeddings = compute_sentence_embeddings(sentences)

    # Compute pairwise sentence similarities
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        sentence_score = util.pytorch_cos_sim(sentence_embeddings[i], sentence_embeddings).sum().item()
        sentence_scores[sentence] = sentence_score / len(sentences)  # Normalize by number of sentences

    # Get the top `summary_ratio`% of sentences with the highest scores
    num_summary_sentences = max(1, int(len(sentences) * summary_ratio))
    summary_sentences = heapq.nlargest(num_summary_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary


def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
    c = 0
    text = ""

    while c < len(pdf_reader.pages):
        pageObj = pdf_reader.pages[c]
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

text_to_process = ""

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
    text_to_process = st.text_area("Enter your paragraph to summarize here:", height=120)

# Button to process the text
if st.button("Generate Summary"):
    if text_to_process:
        with st.spinner("Generating summary..."):
            # extractive_summary = generate_extractive_summary(text_to_process)
            try:
                extractive_summary = generate_extractive_summary(text_to_process)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                extractive_summary = None
            try:
                abstractive_summary = generate_abstractive_summary(text_to_process)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                abstractive_summary = None

        # try:
        #     extractive_summary = generate_extractive_summary(text_to_process)
        # except Exception as e:
        #     st.error(f"An error occurred: {e}")
        #     extractive_summary = None

        if extractive_summary:
            st.subheader("Extractive Summary:")
            st.write(extractive_summary)
        else:
            st.error(f"Error: while getting the extractive summary.")

        # try:
        #     abstractive_summary = generate_abstractive_summary(text_to_process)
        # except Exception as e:
        #     st.error(f"An error occurred: {e}")
        #     abstractive_summary = None

        if abstractive_summary:
            st.subheader("Abstractive Summary:")
            st.write(abstractive_summary)
        else:
            st.error(f"Error: while getting the abstractive summary.")
    else:
        st.error("Please enter some text before processing.")

