import nltk
nltk.download('punkt')  # Download required tokenizer

from nltk.tokenize import sent_tokenize  # ✅ Add this
import re  # ✅ Add this

sample_text = """
IN THE SUPREME COURT OF INDIA

CIVIL APPELLATE JURISDICTION

CIVIL APPEAL NO. 1234 OF 2020

ABC Corporation Ltd. ...Appellant

Vs.

XYZ Enterprises Pvt. Ltd. ...Respondent

JUDGMENT

This appeal arises out of the judgment dated 10th March 2020 passed by the High Court. The High Court held that the agreement between the parties was valid and enforceable. The appellant challenged the same, arguing that the contract was void due to lack of consideration. After hearing both sides, this Court holds that the agreement was valid, and the appeal is accordingly dismissed.
"""

def preprocess(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    return sentences

if __name__ == "__main__":
    sentences = preprocess(sample_text)
    print("\n--- Tokenized Sentences ---\n")
    for s in sentences:
        print(s)
