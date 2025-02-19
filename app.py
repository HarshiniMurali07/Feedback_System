import streamlit as st
import pandas as pd
import random
from datetime import datetime
from textblob import TextBlob

# Simulated feedback data
feedback_data = []

# User authentication system (Admin, Staff, Public Users)
user_roles = {"admin": "admin123", "staff": "staff123"}

def authenticate_user():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username", "")
    password = st.sidebar.text_input("Password", "", type="password")
    if st.sidebar.button("Login"):
        if username in user_roles and user_roles[username] == password:
            return username
        else:
            st.sidebar.error("Invalid username or password")
    return None

def analyze_sentiment(feedback_text):
    analysis = TextBlob(feedback_text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def collect_feedback():
    st.title("Patient Feedback Form")
    name = st.text_input("Your Name")
    department = st.selectbox("Department", ["Cardiology", "Neurology", "General Medicine", "Emergency", "Others"])
    rating = st.slider("Rate your experience (1-5)", 1, 5, 3)
    feedback_text = st.text_area("Additional Comments")
    
    if st.button("Submit Feedback"):
        sentiment = analyze_sentiment(feedback_text)
        feedback_entry = {
            "name": name,
            "department": department,
            "rating": rating,
            "feedback_text": feedback_text,
            "sentiment": sentiment,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        feedback_data.append(feedback_entry)
        st.success(f"Thank you for your feedback! Sentiment detected: {sentiment}")
        
        if rating > 3:
            st.markdown("[Click here to leave a Google Review](https://www.google.com)")

def show_dashboard():
    st.title("Admin & Staff Dashboard")
    if not feedback_data:
        st.warning("No feedback received yet.")
        return
    
    df = pd.DataFrame(feedback_data)
    st.write("### Feedback Summary")
    st.dataframe(df)
    
    st.write("### Department-wise Feedback")
    dept_counts = df["department"].value_counts()
    st.bar_chart(dept_counts)
    
    st.write("### Rating Distribution")
    rating_counts = df["rating"].value_counts()
    st.bar_chart(rating_counts)
    
    st.write("### Sentiment Analysis")
    sentiment_counts = df["sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

def main():
    st.sidebar.title("Feedback Management System")
    mode = st.sidebar.radio("Navigation", ["Submit Feedback", "Admin Dashboard"])
    
    logged_in_user = authenticate_user()
    
    if mode == "Submit Feedback":
        collect_feedback()
    elif mode == "Admin Dashboard":
        if logged_in_user == "admin" or logged_in_user == "staff":
            show_dashboard()
        else:
            st.warning("Admin or Staff login required to access the dashboard.")

if __name__ == "__main__":
    main()
