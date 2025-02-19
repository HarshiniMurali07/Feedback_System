import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

MODEL_PATH = os.path.dirname(os.path.abspath(__file__))

# Load trained models
sentiment_model = joblib.load(os.path.join(MODEL_PATH, "naive_bayes_tuned_model.pkl"))
fake_review_model = joblib.load(os.path.join(MODEL_PATH, "fake_review_detection_model.pkl"))
forecast_model = joblib.load(os.path.join(MODEL_PATH, "manual_arima_feedback_forecasting.pkl"))

# Sidebar Navigation
st.sidebar.title("🔍 Feedback Management System")
page = st.sidebar.radio("Go to", ["Sentiment Analysis", "Fake Review Detection", "Future Trend Prediction", "Download Reports"])

# Sentiment Analysis Page
if page == "Sentiment Analysis":
    st.title("📊 Sentiment Analysis")
    user_input = st.text_area("Enter your feedback:")
    if st.button("Analyze Sentiment"):
        sentiment = sentiment_model.predict([user_input])[0]
        st.subheader(f"Predicted Sentiment: {sentiment}")
        
        # Word Cloud Visualization
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(user_input)
        st.image(wordcloud.to_array(), use_column_width=True)

# Fake Review Detection Page
elif page == "Fake Review Detection":
    st.title("🕵️‍♂️ Fake Review Detection")
    review_input = st.text_area("Enter a review:")
    if st.button("Check if Fake"):
        result = fake_review_model.predict([review_input])[0]
        confidence = np.max(fake_review_model.predict_proba([review_input]))
        st.subheader(f"Predicted: {result} (Confidence: {confidence:.2f})")

# Future Trend Prediction Page
elif page == "Future Trend Prediction":
    st.title("📈 Future Trend Prediction")
    num_days = st.slider("Select number of days to forecast:", 7, 30, 15)
    forecast = forecast_model.forecast(steps=num_days)
    
    # Plot Forecasting Results
    fig, ax = plt.subplots()
    ax.plot(range(num_days), forecast, marker='o', linestyle='dashed')
    ax.set_title("Predicted Future Satisfaction Ratings")
    ax.set_xlabel("Days")
    ax.set_ylabel("Avg Rating")
    st.pyplot(fig)

# Download Reports Page
elif page == "Download Reports":
    st.title("📥 Download Reports")
    st.write("Generate reports for sentiment analysis and trend forecasting.")
    report_data = pd.DataFrame({
        "Feedback": ["Great hospital", "Worst experience", "Friendly staff", "Long waiting time"],
        "Sentiment": ["Positive", "Negative", "Positive", "Negative"]
    })
    st.dataframe(report_data)
    csv = report_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Report as CSV", csv, "report.csv", "text/csv")

st.sidebar.info("Developed with ❤️ using Streamlit")
