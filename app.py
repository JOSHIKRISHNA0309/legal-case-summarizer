import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from heapq import nlargest
import string

nltk.download('punkt')
nltk.download('stopwords')

# --- Summarization Logic ---
def extractive_summary(text, num_sentences=5):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words("english"))
    word_frequencies = {}

    for word in nltk.word_tokenize(text.lower()):
        if word not in stop_words and word not in string.punctuation:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

    sentence_scores = {}
    for sent in sentences:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]

    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    return ' '.join(summary_sentences)

# --- Streamlit UI ---
st.set_page_config(page_title="Legal Case Summarizer", layout="centered")
st.title("🧠 Legal Case Summarizer")

text_input = st.text_area("📜 Paste legal case text here:")

if st.button("Summarize"):
    if text_input.strip():
        summary = extractive_summary(text_input)
        st.subheader("📄 Summary:")
        st.write(summary)
    else:
        st.warning("⚠️ Please enter some text to summarize.")
