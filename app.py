import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import os

# Ensure models are correctly located
MODEL_PATH = os.path.dirname(os.path.abspath(__file__))

# Load trained models
sentiment_model = joblib.load(os.path.join(MODEL_PATH, "naive_bayes_tuned_model.pkl"))
fake_review_model = joblib.load(os.path.join(MODEL_PATH, "fake_review_detection_model.pkl"))
forecast_model = joblib.load(os.path.join(MODEL_PATH, "manual_arima_feedback_forecasting.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_PATH, "label_encoder.pkl"))

# Custom CSS for better UI
def apply_custom_styles():
    st.markdown(
        """
        <style>
        body {background-color: #f0f2f6; font-family: 'Arial', sans-serif;}
        .stButton>button {background-color: #4CAF50; color: white; border-radius: 10px; padding: 10px 20px;}
        .stTextArea>textarea {border-radius: 10px;}
        .stSidebar {background-color: #2e3b4e; color: white;}
        </style>
        """,
        unsafe_allow_html=True
    )

apply_custom_styles()

# Sidebar Navigation
st.sidebar.title("🔍 Feedback Management System")
st.sidebar.image("https://source.unsplash.com/400x200/?feedback", use_column_width=True)
page = st.sidebar.radio("Go to", ["Sentiment Analysis", "Fake Review Detection", "Future Trend Prediction", "Download Reports"])

# Sentiment Analysis Page
if page == "Sentiment Analysis":
    st.title("📊 Sentiment Analysis")
    st.subheader("Analyze customer feedback sentiment")
    user_input = st.text_area("Enter your feedback:")
    if st.button("Analyze Sentiment"):
        prediction = sentiment_model.predict([user_input])
        decoded_prediction = label_encoder.inverse_transform(prediction)
        st.success(f"Predicted Sentiment: {decoded_prediction[0]}")
        
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
        st.success(f"Predicted: {result} (Confidence: {confidence:.2f})")

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
