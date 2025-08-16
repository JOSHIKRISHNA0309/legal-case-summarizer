# -*- coding: utf-8 -*-
"""
Streamlit Web App for Legal Text Summarization
Uses NLTK (for preprocessing) and Sumy (TextRank summarization).
"""

import streamlit as st
import nltk
import re
from nltk.tokenize import sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# Download required NLTK models
nltk.download("punkt")

# ---------------- Preprocessing ----------------
def preprocess(text):
    """
    Preprocess legal text:
    - Removes extra spaces
    - Removes digits (case numbers, years, etc.)
    - Splits into sentences
    """
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+", "", text)
    return sent_tokenize(text)


# ---------------- Summarization ----------------
def extractive_summary(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)


# ---------------- Streamlit UI ----------------
def main():
    st.set_page_config(page_title="Legal Text Summarizer", layout="centered")
    st.title("⚖️ Legal Text Summarizer")
    st.write("Upload or paste legal case documents to generate an **extractive summary**.")

    # User input: either paste text or upload file
    option = st.radio("Choose input method:", ("Paste Text", "Upload File"))

    input_text = ""

    if option == "Paste Text":
        input_text = st.text_area("Enter/Paste Legal Document:", height=250)

    elif option == "Upload File":
        uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
        if uploaded_file is not None:
            input_text = uploaded_file.read().decode("utf-8")

    # Sentence count slider
    sentence_count = st.slider("Number of sentences in summary:", 2, 10, 3)

    if st.button("Generate Summary"):
        if input_text.strip() == "":
            st.warning("⚠️ Please provide some legal text to summarize.")
        else:
            st.subheader("📝 Preprocessed Sentences:")
            sentences = preprocess(input_text)
            st.write(sentences)

            st.subheader("📑 Extractive Summary:")
            summary = extractive_summary(input_text, sentence_count)
            st.success(summary)


if __name__ == "__main__":
    main()
