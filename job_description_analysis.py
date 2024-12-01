import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from spacy import displacy
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load SpaCy model for NER
nlp = spacy.load("en_core_web_sm")
df=pd.read_csv("job_descriptions.csv")
# Load the dataset (replace with your dataset path)
# df = pd.read_csv("job_postings.csv")  # Adjust file path as needed

# Retain only the important columns
df = df[['Job Id', 'Job Title', 'Job Description', 'Role', 'skills',
         'Responsibilities', 'Qualifications', 'Experience', 'Benefits', 'Company Profile']]

# Drop rows with missing Job Description
df.dropna(subset=['Job Description'], inplace=True)

# Display the first few rows to verify
# df.head()
# Function for text preprocessing
def preprocess_text(text):
    # Tokenize, lowercase, and remove stopwords
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens


# Function to extract skills and qualifications dynamically
def extract_entities(text):
    # Process text using SpaCy
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]

    # Extract relevant entities (e.g., SKILL, DEGREE, ORG)
    relevant_entities = [entity[0] for entity in entities if entity[1] in ['SKILL', 'ORG', 'PRODUCT', 'WORK_OF_ART']]
    return relevant_entities

# Example: Extract entities from a single job description
# example_desc = df.iloc[0]['Job Description']
# extracted_skills = extract_entities(example_desc)
# print("Extracted Skills:", extracted_skills)

# Function to summarize job description dynamically
def summarize_with_tfidf(text, num_sentences=3):
    # Split the text into sentences
    sentences = sent_tokenize(text)

    # Use TF-IDF to rank sentences
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = tfidf_matrix.sum(axis=1).A1

    # Get the top N sentences
    ranked_sentences = sentence_scores.argsort()[-num_sentences:][::-1]
    summary = " ".join([sentences[i] for i in ranked_sentences])
    return summary

# # Example: Summarize a job description
# summary = summarize_with_tfidf(example_desc, num_sentences=3)
# print("Summary:", summary)

# Preprocess job descriptions for classification
def preprocess_for_classification(text):
    tokens = preprocess_text(text)
    return " ".join(tokens)

# Apply preprocessing and store the results using `.loc` to avoid warnings
# df.loc[:, 'processed_description'] = df['Job Description'].apply(preprocess_for_classification)
#
# # Use Job Role as the target variable
# X = df['processed_description']
# y = df['Role']
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Train a Naive Bayes model for job role classification
# model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
# model.fit(X_train, y_train)

# Preprocess the 'Job Description' column for classification
df['processed_description'] = df['Job Description'].apply(preprocess_for_classification)

# Use Job Role as the target variable
X = df['processed_description']  # Feature
y = df['Role']  # Target

# Check for any imbalance in the dataset
# print(y.value_counts())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# Explanation:
# - `stratify=y` ensures that the class distribution in the training and testing sets matches the original dataset.

# Train a pipeline with TF-IDF and Multinomial Naive Bayes
model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())

# Fit the model on training data
model.fit(X_train, y_train)

# Evaluate the model
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# Ensure the results are meaningful
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred))


# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model
with open('job_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)


# Load the saved model
with open('job_classifier.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Function to predict the job role
def predict_job_role(description):
    preprocessed_desc = preprocess_for_classification(description)
    return loaded_model.predict([preprocessed_desc])[0]

# # Example: Predict a job role
# job_desc_input = "We are looking for a highly motivated marketing professional with experience in digital advertising and SEO."
# predicted_role = predict_job_role(job_desc_input)
# print("Predicted Role:", predicted_role)


# Function to combine summarization, skills extraction, and prediction
def analyze_job_description(description):
    # Summarize
    summary = summarize_with_tfidf(description, num_sentences=3)

    # Extract skills/entities
    skills = extract_entities(description)

    # Predict job role
    predicted_role = predict_job_role(description)

    # Return combined result
    return {
        "Summary": summary,
        "Extracted Skills": skills,
        "Predicted Role": predicted_role
    }

# # Example: Analyze a job description
# example_desc = df.iloc[0]['Job Description']
# result = analyze_job_description(example_desc)
# print("Analysis Result:", result)
