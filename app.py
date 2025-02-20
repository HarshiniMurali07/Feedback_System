import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import os

# Ensure models are correctly located
MODEL_PATH = os.path.dirname(os.path.abspath(__file__))

# Load trained models
sentiment_model = joblib.load(os.path.join(MODEL_PATH, "naive_bayes_tuned_model.pkl"))
tfidf_vectorizer = joblib.load(os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl"))
fake_review_model = joblib.load(os.path.join(MODEL_PATH, "fake_review_detection_model.pkl"))
forecast_model = joblib.load(os.path.join(MODEL_PATH, "manual_arima_feedback_forecasting.pkl"))

# Custom CSS for Modern UI
def apply_custom_styles():
    st.markdown(
        """
        <style>
        body {background-color: #e3f2fd; font-family: 'Arial', sans-serif;}
        .main-container {background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);}
        .stButton>button {background-color: #0288d1; color: white; border-radius: 10px; padding: 10px 20px; border: none;}
        .stTextArea>textarea {border-radius: 10px; border: 1px solid #0288d1;}
        .stSidebar {background-color: #1e3a56; color: white; padding: 20px;}
        .header-text {color: #0288d1; text-align: center; font-size: 26px; font-weight: bold;}
        .sub-header {color: #1565c0; text-align: center; font-size: 20px; font-weight: bold;}
        </style>
        """,
        unsafe_allow_html=True
    )

apply_custom_styles()

# Home Page
st.markdown("<div class='header-text'>📝 Welcome to the Feedback Management System</div>", unsafe_allow_html=True)
st.image("https://source.unsplash.com/800x300/?feedback,technology", use_column_width=True)
st.write("This system allows you to analyze feedback using Sentiment Analysis, detect Fake Reviews, and predict future trends.")

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
st.sidebar.image("https://source.unsplash.com/400x200/?feedback", use_column_width=True)
page = st.sidebar.radio("Choose an Option", ["Home", "Sentiment Analysis", "Fake Review Detection", "Future Trend Prediction", "Download Reports"])

# Sentiment Analysis Page
if page == "Sentiment Analysis":
    st.title("📊 Sentiment Analysis")
    st.subheader("Analyze customer feedback sentiment")
    user_input = st.text_area("Enter your feedback:")
    if st.button("Analyze Sentiment"):
        transformed_text = tfidf_vectorizer.transform([user_input])  # Apply TF-IDF transformation
        prediction = sentiment_model.predict(transformed_text)
        st.subheader(f"Predicted Sentiment: {prediction[0]}")
        st.write(f"Debug: Raw Prediction Output → {prediction}")
        
        # Word Cloud Visualization
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(user_input)
        st.image(wordcloud.to_array(), use_column_width=True)

# Fake Review Detection Page
elif page == "Fake Review Detection":
    st.title("🕵️‍♂️ Fake Review Detection")
    st.subheader("Identify fake customer reviews")
    review_input = st.text_area("Enter a review:")
    if st.button("Check if Fake"):
        result = fake_review_model.predict([review_input])[0]
        confidence = np.max(fake_review_model.predict_proba([review_input]))
        st.subheader(f"Predicted: {result} (Confidence: {confidence:.2f})")
        st.write(f"Debug: Raw Prediction Output → {result}")

# Future Trend Prediction Page
elif page == "Future Trend Prediction":
    st.title("📈 Future Trend Prediction")
    st.subheader("Forecast future satisfaction ratings")
    num_days = st.slider("Select number of days to forecast:", 7, 30, 15)
    forecast = forecast_model.forecast(steps=num_days)
    
    # Plot Forecasting Results
    fig, ax = plt.subplots()
    ax.plot(range(num_days), forecast, marker='o', linestyle='dashed', color='blue')
    ax.set_title("Predicted Future Satisfaction Ratings")
    ax.set_xlabel("Days")
    ax.set_ylabel("Avg Rating")
    st.pyplot(fig)
    st.write(f"Debug: Forecast Output → {forecast}")

# Download Reports Page
elif page == "Download Reports":
    st.title("📥 Download Reports")
    st.subheader("Generate reports for sentiment analysis and trend forecasting.")
    report_data = pd.DataFrame({
        "Feedback": ["Great hospital", "Worst experience", "Friendly staff", "Long waiting time"],
        "Sentiment": ["Positive", "Negative", "Positive", "Negative"]
    })
    st.dataframe(report_data)
    csv = report_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Report as CSV", csv, "report.csv", "text/csv")

st.sidebar.info("Developed with ❤️ using Streamlit")
