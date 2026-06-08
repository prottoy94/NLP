import streamlit as st
import pickle
import nltk
import re
import string
import joblib
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=3000,stop_words='english')

nltk.download('punkt')
nltk.download('stopwords')

clf = pickle.load(open('knn_model.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))
tfidf = joblib.load('tfidf_vectorizer.pkl')

def preprocess(txt):
    txt = str(txt).lower().strip()
    txt = txt.replace("\n"," ").replace("\r"," ").replace("\t"," ")
    txt = txt.replace("’","'").replace("‘","'").replace("“",'"').replace("”",'"')
    txt = re.sub(r'https?\S+\s|www\.\S+\s', '', txt)
    txt = re.sub(r'\S+@\S+\s', '', txt)
    txt = re.sub(r'\S+#\S+\s', '', txt)
    txt = re.sub(r'[^\x00-\x7f]', '', txt)
    
    txt = txt.replace("%"," percent ").replace("$"," dollar ").replace("₹"," rupee ").replace("€"," euro ").replace("£"," pound ").replace("¥"," yen ").replace("₩"," won ").replace("₽"," ruble ").replace("₺"," lira ").replace("₴"," hryvnia ").replace("₦"," naira ").replace("₵"," cedi ").replace("₸"," tenge ").replace("₼"," manat ").replace("₽"," ruble ").replace("₾"," lari ").replace("₿"," bitcoin ")
    txt=re.sub('[%s]' % re.escape(string.punctuation), ' ', txt)
    txt = txt.replace('[math]','').replace('[/math]','')
    txt = txt.replace('000,000,000','b ').replace('000,000','m').replace('000','k')
    txt = re.sub(r'([0-9]+)000000000',' \1b ',txt)
    txt = re.sub(r'([0-9]+)000000',' \1m ',txt)
    txt = re.sub(r'([0-9]+)000',' \1k ',txt)
    
    CONTRACTIONS = {
    # AM
    "ain't":        "am not",
    "i'm":          "i am",

    # ARE
    "aren't":       "are not",
    "you're":       "you are",
    "we're":        "we are",
    "they're":      "they are",
    "who're":       "who are",
    "what're":      "what are",
    "there're":     "there are",

    # IS / HAS
    "isn't":        "is not",
    "he's":         "he is",
    "she's":        "she is",
    "it's":         "it is",
    "that's":       "that is",
    "what's":       "what is",
    "who's":        "who is",
    "where's":      "where is",
    "when's":       "when is",
    "why's":        "why is",
    "how's":        "how is",
    "here's":       "here is",
    "there's":      "there is",
    "this's":       "this is",

    # WAS / WERE
    "wasn't":       "was not",
    "weren't":      "were not",

    # HAVE
    "haven't":      "have not",
    "i've":         "i have",
    "you've":       "you have",
    "we've":        "we have",
    "they've":      "they have",
    "could've":     "could have",
    "should've":    "should have",
    "would've":     "would have",
    "might've":     "might have",
    "must've":      "must have",
    "who've":       "who have",

    # HAD / WOULD
    "hadn't":       "had not",
    "i'd":          "i would",
    "you'd":        "you would",
    "he'd":         "he would",
    "she'd":        "she would",
    "we'd":         "we would",
    "they'd":       "they would",
    "who'd":        "who would",
    "what'd":       "what did",
    "it'd":         "it would",

    # HAS
    "hasn't":       "has not",

    # WILL
    "won't":        "will not",
    "i'll":         "i will",
    "you'll":       "you will",
    "he'll":        "he will",
    "she'll":       "she will",
    "we'll":        "we will",
    "they'll":      "they will",
    "who'll":       "who will",
    "that'll":      "that will",
    "it'll":        "it will",

    # DO / DOES / DID
    "don't":        "do not",
    "doesn't":      "does not",
    "didn't":       "did not",

    # COULD / SHOULD / WOULD / MIGHT / MUST
    "couldn't":     "could not",
    "shouldn't":    "should not",
    "wouldn't":     "would not",
    "mightn't":     "might not",
    "mustn't":      "must not",
    "couldn't've":  "could not have",
    "shouldn't've": "should not have",
    "wouldn't've":  "would not have",

    # CAN
    "can't":        "cannot",
    "can't've":     "cannot have",

    # NEED / DARE
    "needn't":      "need not",
    "needn't've":   "need not have",
    "daren't":      "dare not",

    # MISC
    "'cause":       "because",
    "let's":        "let us",
    "ma'am":        "madam",
    "o'clock":      "of the clock",
    "o'er":         "over",
    "ne'er":        "never",
    "e'er":         "ever",
    "y'all":        "you all",
    "y'all'd":      "you all would",
    "y'all're":     "you all are",
    "y'all've":     "you all have",
    "not've":       "not have",
    "how'd":        "how did",
    "how'd'y":      "how do you",
    "how'll":       "how will",
    "i'd've":       "i would have",
    "it'd've":      "it would have",
    "she'd've":     "she would have",
    "he'd've":      "he would have",
    "we'd've":      "we would have",
    "they'd've":    "they would have",
    "when'd":       "when did",
    "where'd":      "where did",
    "why'd":        "why did",
    "who'd've":     "who would have",
    "that'd":       "that would",
    "that'd've":    "that would have",
    "there'd":      "there would",
    "there'd've":   "there would have",
    "here'd":       "here would",
    "how's":        "how is",
    "i'm":          "i am",
    "you've":       "you have",
    "we've":        "we have",
    "they've":      "they have",
    "who've":       "who have",
    "would've":     "would have",
    "should've":    "should have",
    "could've":     "could have",
    "must've":      "must have",    
    }
    
    txt_dectracted = []
    
    for word in txt.split():
        if word in CONTRACTIONS:
            txt_dectracted.append(CONTRACTIONS[word])
        else:
            txt_dectracted.append(word)
            
    txt = " ".join(txt_dectracted)
    txt = txt.replace("'ve"," have").replace("'re"," are").replace("'ll"," will").replace("'d"," would").replace("'s"," is").replace("n't"," not")
    
    txt = BeautifulSoup(txt, "html.parser")
    txt = txt.get_text()
    
    pattern = re.compile(r'\W')
    txt = re.sub(pattern, ' ', txt).strip()
    
    return txt

def main():
    st.title("Resume Screening Application")
    st.write("Upload a resume to predict its category.")
    
    uploaded_file = st.file_uploader("Choose a resume file", type=["txt", "pdf", "docx"])
    
    if uploaded_file is not None:
        try:
            resume_text = uploaded_file.read().decode('utf-8')
        except Exception as e:
            resume_text = uploaded_file.read().decode('latin-1')
        
        cleaned_text = preprocess(resume_text)  # CHANGED: cleanResume → preprocess

        # Vectorize the cleaned text using the same TF-IDF vectorizer used during training
        vectorized_text = tfidf.transform([cleaned_text])
        
        # Convert sparse matrix to dense
        vectorized_text = vectorized_text.toarray()

        # Prediction - use your trained KNN model (not svc_model)
        predicted_category = clf.predict(vectorized_text)  # CHANGED: svc_model → clf

        # get name of predicted category
        predicted_category_name = le.inverse_transform(predicted_category)
        print(predicted_category_name[0])  # Print the predicted category name
        st.write(f"Predicted Category: {predicted_category_name[0]}")
    
if __name__ == "__main__":
    main()
     