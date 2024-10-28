import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration
import torch
import re
from collections import Counter


@st.cache_resource
def load_summarization_model():
    model_name = "t5-small" 
    summarizer_tokenizer = AutoTokenizer.from_pretrained(model_name)
    summarizer_model = T5ForConditionalGeneration.from_pretrained(model_name)
    return summarizer_tokenizer, summarizer_model
# Initialize BERT model
@st.cache_resource
def load_model():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Database operations
def init_db():
    conn = sqlite3.connect('professor_reviews.db')
    c = conn.cursor()
    
    # Create professors table
    c.execute('''
        CREATE TABLE IF NOT EXISTS professors
        (id TEXT PRIMARY KEY, name TEXT, department TEXT)
    ''')
    
    # Create reviews table
    c.execute('''
        CREATE TABLE IF NOT EXISTS reviews
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         prof_id TEXT,
         review_text TEXT,
         rating REAL,
         timestamp TEXT,
         FOREIGN KEY (prof_id) REFERENCES professors (id))
    ''')
    
    # Insert dummy professors if they don't exist
    dummy_professors = [
        ('PROF001', 'Dr. Sarah Smith', 'Computer Science'),
        ('PROF002', 'Dr. John Davis', 'Physics'),
        ('PROF003', 'Dr. Maria Garcia', 'Mathematics'),
        ('PROF004', 'Dr. James Wilson', 'Chemistry'),
        ('PROF005', 'Dr. Emily Brown', 'Biology')
    ]
    
    c.executemany('''
        INSERT OR IGNORE INTO professors (id, name, department)
        VALUES (?, ?, ?)
    ''', dummy_professors)
    
    conn.commit()
    conn.close()

class ReviewAnalyzer:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
    
    def analyze_single_review(self, review):
        max_length = 512
        review = ' '.join(review.split()[:max_length])
        
        inputs = self.tokenizer(review, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.softmax(dim=1)
            
        predicted_rating = torch.argmax(scores) + 1
        rating_10_scale = (predicted_rating.item() - 1) * 2.5
        return rating_10_scale

def get_professor_details(prof_id):
    conn = sqlite3.connect('professor_reviews.db')
    c = conn.cursor()
    c.execute('SELECT * FROM professors WHERE id = ?', (prof_id,))
    result = c.fetchone()
    conn.close()
    return result if result else None

def save_review(prof_id, review_text, rating):
    conn = sqlite3.connect('professor_reviews.db')
    c = conn.cursor()
    # Store timestamp in ISO format
    timestamp = datetime.now().isoformat()
    c.execute('''
        INSERT INTO reviews (prof_id, review_text, rating, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (prof_id, review_text, rating, timestamp))
    conn.commit()
    conn.close()

def get_all_reviews():
    conn = sqlite3.connect('professor_reviews.db')
    reviews_df = pd.read_sql_query('''
        SELECT r.*, p.name as professor_name, p.department
        FROM reviews r
        JOIN professors p ON r.prof_id = p.id
        ORDER BY r.timestamp DESC
    ''', conn)
    conn.close()
    
    # Convert timestamp strings to pandas datetime with format inference
    reviews_df['timestamp'] = pd.to_datetime(reviews_df['timestamp'], format='mixed')
    return reviews_df

def main():
    st.set_page_config(page_title="Professor Review System", layout="wide")
    
    # Initialize database
    init_db()
    
    # Load model
    tokenizer, model = load_model()
    analyzer = ReviewAnalyzer(tokenizer, model)
    
    # Check if we're in admin mode using the new query_params
    path = st.query_params.get("page", "main")
    
    if path == "admin":
        admin_page()
    else:
        user_page(analyzer)

def user_page(analyzer):
    st.title("Professor Review System")
    
    # Get list of professors
    conn = sqlite3.connect('professor_reviews.db')
    professors_df = pd.read_sql_query('SELECT * FROM professors', conn)
    conn.close()
    
    # Create selectbox for professors
    selected_prof_id = st.selectbox(
        "Select Professor",
        professors_df['id'].tolist(),
        format_func=lambda x: f"{professors_df[professors_df['id']==x]['name'].iloc[0]} ({professors_df[professors_df['id']==x]['department'].iloc[0]})"
    )
    
    # Show professor details
    prof_details = get_professor_details(selected_prof_id)
    if prof_details:
        st.write(f"Department: {prof_details[2]}")
    
    # Review input
    review_text = st.text_area("Write your review", height=150)
    
    if st.button("Submit Review"):
        if review_text.strip():
            # Analyze review
            rating = analyzer.analyze_single_review(review_text)
            
            # Save to database
            save_review(selected_prof_id, review_text, rating)
            
            st.success("Review submitted successfully!")
            st.write(f"AI-Generated Rating: {rating:.1f}/10")
        else:
            st.error("Please write a review before submitting.")

def summarize_reviews_bart(reviews, tokenizer, model):
    reviews_text = ' '.join(reviews)  # Combine all reviews for input
    inputs = tokenizer(reviews_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=40, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def admin_page():
    st.title("Admin Dashboard - Professor Reviews")
    
    # Get all reviews
    reviews_df = get_all_reviews()
    
    # Display summary statistics
    st.header("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Reviews", len(reviews_df))
    with col2:
        avg_rating = reviews_df['rating'].mean()
        st.metric("Average Rating", f"{avg_rating:.1f}/10")
    with col3:
        # Calculate reviews from the last 7 days
        week_ago = pd.Timestamp.now() - pd.Timedelta(days=7)
        recent_reviews = len(reviews_df[reviews_df['timestamp'] > week_ago])
        st.metric("Reviews This Week", recent_reviews)
    
    # Display reviews by professor
    st.header("Reviews by Professor")
    
    summarizer_tokenizer, summarizer_model = load_summarization_model()

    for prof_id in reviews_df['prof_id'].unique():
        prof_reviews = reviews_df[reviews_df['prof_id'] == prof_id]
        prof_name = prof_reviews['professor_name'].iloc[0]
        dept = prof_reviews['department'].iloc[0]
        
        # Generate a summary for all reviews for this professor
        summary_text = summarize_reviews_bart(prof_reviews['review_text'].tolist(), summarizer_tokenizer, summarizer_model)
        
        with st.expander(f"{prof_name} ({dept}) - Avg Rating: {prof_reviews['rating'].mean():.1f}/10"):
            st.write(f"Summary of teaching style: {summary_text}")
            display_df = prof_reviews.copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(
                display_df[['timestamp', 'review_text', 'rating']]
                .sort_values('timestamp', ascending=False)
                .reset_index(drop=True)
            )
            
def summarize_reviews(reviews):
    # Preprocess reviews to extract words
    reviews_text = ' '.join(reviews)
    words = re.findall(r'\w+', reviews_text.lower())
    
    # Common words analysis (excluding stopwords)
    stopwords = set(['the', 'and', 'of', 'to', 'a', 'is', 'in', 'that', 'it', 'for', 'with', 'as', 'on', 'are', 'by'])
    filtered_words = [word for word in words if word not in stopwords]
    common_words = Counter(filtered_words).most_common(5)
    
    # Join common words into a readable summary
    keywords = ', '.join(word for word, _ in common_words)
    return f"Focuses on {keywords}"


if __name__ == "__main__":
    main()