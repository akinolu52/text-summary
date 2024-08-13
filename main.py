from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import re
import heapq
from nltk.tokenize import sent_tokenize, word_tokenize

app = Flask(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2')


def clean_text(text):
    text = text.strip()

    # remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)

    # remove \n and quotes like “, ‘, ”
    text = re.sub(r'[\n“”‘’]', '', text)

    return text


def compute_sentence_embeddings(sentences):
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return embeddings


def summarize_text_with_transformer(text, summary_ratio=0.3):
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


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/process', methods=['POST'])
def process_text():
    data = request.json
    text = data.get('paragraph')

    if not text:
        return jsonify({'error': 'Missing text or metric'}), 400

    if not summarize_text_with_transformer:
        return jsonify({'error': 'Invalid metric'}), 400

    summary = summarize_text_with_transformer(text)
    return jsonify({
        'summary': summary,
    })


if __name__ == '__main__':
    app.run(debug=True)
