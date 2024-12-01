# Streamlit App for Job Description Analysis

import streamlit as st
import pandas as pd
import pickle
import spacy
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load the trained model
with open('job_classifier.pkl', 'rb') as f:
    model = pickle.load(f)


# Preprocess for classification
def preprocess_for_classification(text):
    tokens = text.lower().split()
    return " ".join(tokens)


# Summarize using TF-IDF
def summarize_with_tfidf(text, num_sentences=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text  # Return the original text if it has fewer sentences

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = tfidf_matrix.sum(axis=1).A1
    ranked_sentences = sentence_scores.argsort()[-num_sentences:][::-1]
    summary = " ".join([sentences[i] for i in ranked_sentences])
    return summary


# Extract skills/entities using SpaCy
def extract_entities(text):
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    relevant_entities = [entity[0] for entity in entities if entity[1] in ['SKILL', 'ORG', 'PRODUCT', 'WORK_OF_ART']]
    return relevant_entities


# Predict job role
def predict_job_role(description):
    preprocessed_desc = preprocess_for_classification(description)
    return model.predict([preprocessed_desc])[0]


# Analyze a single job description
def analyze_job_description(description):
    summary = summarize_with_tfidf(description, num_sentences=3)
    skills = extract_entities(description)
    predicted_role = predict_job_role(description)
    return {
        "Summary": summary,
        "Extracted Skills": skills,
        "Predicted Role": predicted_role
    }


# Streamlit app setup
st.title("Job Description Analyzer")

st.markdown("""
This tool analyzes job descriptions to:
- Summarize the content.
- Extract key skills and entities.
- Predict the most likely job role.
""")

# Single job description analysis
st.header("Single Job Description Analysis")
job_description = st.text_area("Enter a Job Description", height=300)

if st.button("Analyze Job Description"):
    if not job_description.strip():
        st.warning("Please enter a job description to analyze.")
    else:
        results = analyze_job_description(job_description)
        st.subheader("Analysis Results")
        st.write("**Predicted Job Role:**", results["Predicted Role"])
        st.write("**Summary:**", results["Summary"])
        st.write("**Extracted Skills/Entities:**", results["Extracted Skills"])

# Batch analysis with file upload
st.header("Batch Analysis (File Upload)")
uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])

if uploaded_file:
    # Load the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    if 'Job Description' not in df.columns:
        st.error("The uploaded file must have a 'Job Description' column.")
    else:
        st.success("File uploaded successfully!")

        # Initialize batch results
        batch_results = []
        for index, row in df.iterrows():
            description = row['Job Description']
            analysis = analyze_job_description(description)
            batch_results.append({
                "Job Description": description,
                "Summary": analysis["Summary"],
                "Extracted Skills": analysis["Extracted Skills"],
                "Predicted Role": analysis["Predicted Role"]
            })

        # Convert results to DataFrame
        results_df = pd.DataFrame(batch_results)
        st.subheader("Batch Analysis Results")
        st.dataframe(results_df)

        # Provide a download option for the results
        st.sidebar.markdown("### Download Results")
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="analyzed_job_descriptions.csv",
            mime="text/csv"
        )
