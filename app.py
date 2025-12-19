import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx2txt


nltk.download('stopwords')

# --------- DATA ----------
data = {
    "name": [
        "Amit Sharma",
        "Neha Verma",
        "Rahul Singh",
        "Priya Patel",
        "Karan Mehta",
        "Sonal Iyer",
        "Arjun Rao"
    ],
    "resume_text": [
        "Python developer with experience in machine learning, data analysis, SQL and statistics",
        "Java developer with knowledge of Spring Boot, Hibernate, MySQL and backend development",
        "Data analyst skilled in Python, SQL, Excel, Power BI and machine learning basics",
        "Frontend developer with skills in HTML, CSS, JavaScript, React and UI design",
        "Software engineer experienced in Python, Django, REST APIs, SQL and cloud basics",
        "AI engineer skilled in deep learning, NLP, Python",
        "Cloud engineer with AWS, Docker, Linux, Python"
    ]
}

df = pd.DataFrame(data)

# --------- CLEANING ----------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['cleaned_resume'] = df['resume_text'].apply(clean_text)

# --------- VECTORIZATION ----------
vectorizer = TfidfVectorizer()
resume_vectors = vectorizer.fit_transform(df['cleaned_resume'])

# --------- UI ----------
st.title("🤖 Smart Resume Screening with AI")
st.subheader("📄 Upload Resumes")

uploaded_files = st.file_uploader(
    "Upload PDF or DOCX resumes",
    type=["pdf", "docx"],
    accept_multiple_files=True
)
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    elif file.name.endswith(".docx"):
        return docx2txt.process(file)

import re

def split_applicants(text):
    parts = re.split(r'(B\.Tech|B\.E|MCA|M\.Tech|BCA|M\.Sc)', text)
    applicants = []

    current = ""
    for part in parts:
        current += " " + part
        if len(current.strip()) > 150:
            applicants.append(current.strip())
            current = ""

    if current.strip():
        applicants.append(current.strip())

    return applicants



st.write("Enter job requirements below:")

job_description = st.text_area(
    "Job Description",
    placeholder="Example: Python, Machine Learning, SQL, Data Analysis"
)

if st.button("🔍 Find Best Candidates"):

    if uploaded_files:
        names = []
        texts = []

        for file in uploaded_files:
            full_text = extract_text(file)
            applicant_blocks = split_applicants(full_text)

            for i, block in enumerate(applicant_blocks):
                names.append(f"{file.name}_candidate_{i+1}")
                texts.append(block)

        df = pd.DataFrame({
            "name": names,
            "resume_text": texts
        })

        df['cleaned_resume'] = df['resume_text'].apply(clean_text)

        vectorizer = TfidfVectorizer()
        resume_vectors = vectorizer.fit_transform(df['cleaned_resume'])

    else:
        st.warning("Please upload at least one resume.")
        st.stop()

    if job_description.strip() == "":
        st.warning("Please enter job requirements.")
    else:
        clean_job = clean_text(job_description)
        job_vector = vectorizer.transform([clean_job])
        scores = cosine_similarity(job_vector, resume_vectors)

        df['match_score'] = scores[0]
        ranked = df.sort_values(by='match_score', ascending=False)

        st.subheader("📊 Ranked Candidates")
        st.dataframe(ranked[['name', 'match_score']])
