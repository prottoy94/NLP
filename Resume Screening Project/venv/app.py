import PyPDF2  # Add this at the top with other imports
import streamlit as st
import pickle
import nltk
import re
import string
import joblib
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

clf = pickle.load(open('knn_model.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))
tfidf = joblib.load('tfidf_vectorizer.pkl')

def preprocess(txt):
    txt = str(txt).lower().strip()
    
    # Common skill section headers
    skill_headers = [
        r'skills?:', r'skills summary:', r'core skills?:', r'key skills?:',
        r'technical skills?:', r'tech skills?:', r'professional skills?:',
        r'expertise:', r'areas of expertise:', r'core competencies:',
        r'competencies:', r'technologies?:', r'technology stack:',
        r'tech stack:', r'programming languages?:', r'languages:',
        r'tools:', r'tools & technologies:', r'technical expertise:',
        r'technical proficiencies:', r'proficiencies:', r'experience:',
        r'familiar with:', r'proficient in:', r'skilled in:',
        r'hands-on experience:', r'working knowledge:', r'platforms?:',
        r'frameworks?:', r'databases?:', r'cloud platforms?:',
        r'devops tools?:', r'development tools?:', r'soft skills?:',
        r'interpersonal skills?:', r'leadership skills?:', r'management skills?:',
        r'communication skills?:', r'certifications?:', r'languages spoken:',
        r'foreign languages:', r'skills & abilities:', r'abilities:',
        r'qualifications:', r'technical background:', r'core qualifications:',
        r'areas of proficiency:', r'skillset:', r'skill set:', r'key competencies:',
        r'main skills:', r'primary skills:', r'specialties:',
        r'areas of specialization:', r'expertise areas:'
    ]
    
    lines = txt.split('\n')
    in_skills_section = False
    skills_lines = []
    
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        # FIX 1: Use re.search instead of re.match
        for header in skill_headers:
            if re.search(header, line_lower):  # Changed from re.match
                in_skills_section = True
                content = re.sub(header, '', line_lower).strip()
                if content:
                    skills_lines.append(content)
                break
        
        if in_skills_section:
            next_section = re.match(r'(education|experience|work|project|certification|language|profile|summary|objective):', line_lower)
            if next_section and i > 0 and skills_lines:
                in_skills_section = False
                break
            
            if line_lower and not re.match(r'^[\d\-\*•\s]+$', line_lower):
                # FIX 2: Use re.search here too
                if not any(re.search(header, line_lower) for header in skill_headers):  # Changed from re.match
                    skills_lines.append(line_lower)
    
    # FIX 3: Add keyword-based extraction as fallback
    if skills_lines:
        txt = ' '.join(skills_lines)
    else:
        # Keyword-based extraction for resumes without clear skills sections
        tech_keywords = [
            'python', 'java', 'javascript', 'typescript', 'sql', 'nosql',
            'react', 'angular', 'vue', 'node', 'django', 'flask', 'spring',
            'c++', 'c#', 'php', 'ruby', 'go', 'machine learning', 
            'deep learning', 'tensorflow', 'pytorch', 'keras', 'pandas', 
            'numpy', 'scikit-learn', 'matplotlib', 'aws', 'azure', 'gcp',
            'docker', 'kubernetes', 'jenkins', 'git', 'agile', 'scrum',
            'html', 'css', 'bootstrap', 'rest api', 'graphql',
            'leadership', 'project management', 'team management'
        ]
        
        keyword_lines = []
        for line in lines:
            line_lower = line.lower().strip()
            if len(line_lower) < 3:
                continue
            for keyword in tech_keywords:
                if keyword in line_lower:
                    keyword_lines.append(line_lower)
                    break
        
        if keyword_lines:
            txt = ' '.join(keyword_lines)
            if len(txt) > 2000:
                txt = txt[:2000]
        else:
            txt = txt[:2000]  # Limit to first 2000 characters if no skills found
    
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

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""
    
def main():
    st.title("Resume Screening Application")
    st.write("Upload a resume to predict its category.")
    
    uploaded_file = st.file_uploader("Choose a resume file", type=["txt", "pdf", "docx"])
    
    if uploaded_file is not None:
        resume_text = ""
        
        # FIX: Handle PDF files properly
        if uploaded_file.type == "application/pdf":
            with st.spinner("Extracting text from PDF..."):
                resume_text = extract_text_from_pdf(uploaded_file)
            if not resume_text:
                st.error("Could not extract text from PDF. The file might be scanned or image-based.")
                st.stop()
        else:
            # Handle TXT files
            try:
                resume_text = uploaded_file.read().decode('utf-8')
            except Exception as e:
                try:
                    resume_text = uploaded_file.read().decode('latin-1')
                except Exception as e2:
                    st.error(f"Error reading file: {str(e2)}")
                    st.stop()
        
        if resume_text:
            cleaned_text = preprocess(resume_text)  # ← FIXED INDENTATION
            
            # Vectorize the cleaned text
            vectorized_text = tfidf.transform([cleaned_text])
            
            # Convert sparse matrix to dense
            vectorized_text = vectorized_text.toarray()
            
            # Prediction
            predicted_category = clf.predict(vectorized_text)
            
            # Get name of predicted category
            predicted_category_name = le.inverse_transform(predicted_category)
            print(predicted_category_name[0])
            st.success(f"Predicted Category: **{predicted_category_name[0]}**")
    
if __name__ == "__main__":
    main()
     