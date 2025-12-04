# Core Packages
import streamlit as st 
import altair as alt
import plotly.express as px 

# EDA Packages
import pandas as pd 
import numpy as np 
from datetime import datetime

# Utils
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "emotion_classifier_pipe_lr.pkl")

# Cached model loading
@st.cache_resource
def load_model():
    return joblib.load(open(MODEL_PATH, "rb"))

pipe_lr = load_model()

# Track Utils
from track_utils import (
    create_page_visited_table,
    add_page_visited_details,
    view_all_page_visited_details,
    add_prediction_details,
    view_all_prediction_details,
    create_emotionclf_table
)

# Prediction Functions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {
    "anger":"😠","disgust":"🤮", "fear":"😨😱", 
    "happy":"🤗", "joy":"😂", "neutral":"😐", 
    "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"
}

# Main Application
def main():
    st.set_page_config(page_title="Emotion Classifier", page_icon="🤖", layout="wide")
    st.title("🤖 Emotion Classifier App")
    
    menu = ["Home 🏠", "Monitor 📊", "About ℹ️"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    create_page_visited_table()
    create_emotionclf_table()
    
    if choice.startswith("Home"):
        add_page_visited_details("Home", datetime.now())
        st.subheader("Emotion Predictor")
        
        raw_text = st.text_area("Type or paste text here to analyze emotions:")
        
        if raw_text.strip():
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)[0]
            top_indices = np.argsort(probability)[::-1][:3]  # Top 3 predictions
            top_emotions = pipe_lr.classes_[top_indices]
            top_probs = probability[top_indices]
            
            add_prediction_details(raw_text, prediction, np.max(probability), datetime.now())
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success("Original Text")
                st.write(raw_text)
                
                st.success("Top Prediction")
                emoji_icon = emotions_emoji_dict.get(prediction, "")
                st.write(f"{prediction}: {emoji_icon}")
                st.progress(float(np.max(probability)))
                
                st.markdown("**Top 3 Predictions:**")
                for emo, prob in zip(top_emotions, top_probs):
                    st.write(f"{emo}: {prob:.2f} {emotions_emoji_dict.get(emo,'')}")
            
            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame({
                    "emotions": pipe_lr.classes_,
                    "probability": probability
                })
                fig = alt.Chart(proba_df).mark_bar().encode(
                    x='emotions',
                    y='probability',
                    color=alt.condition(
                        alt.datum.probability == np.max(probability),
                        alt.value("orange"),  # Highlight top prediction
                        'emotions'
                    ),
                    tooltip=['emotions','probability']
                )
                st.altair_chart(fig, use_container_width=True)
        else:
            st.info("Please enter some text to get predictions.")
    
    elif choice.startswith("Monitor"):
        add_page_visited_details("Monitor", datetime.now())
        st.subheader("Monitor Dashboard")
        
        # Page Metrics
        with st.expander("Page Metrics"):
            page_visits = pd.DataFrame(view_all_page_visited_details(), columns=['Pagename','Time_of_Visit'])
            st.dataframe(page_visits)
            
            pg_count = page_visits['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(x='Pagename', y='Counts', color='Pagename', tooltip=['Pagename','Counts'])
            st.altair_chart(c, use_container_width=True)
            
            p = px.pie(pg_count, values='Counts', names='Pagename')
            st.plotly_chart(p, use_container_width=True)
        
        # Emotion Classifier Metrics
        with st.expander("Emotion Classifier Metrics"):
            df_emotions = pd.DataFrame(view_all_prediction_details(), columns=['Rawtext','Prediction','Probability','Time_of_Visit'])
            st.dataframe(df_emotions)
            
            st.markdown("**Prediction Distribution:**")
            prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(
                x='Prediction', 
                y='Counts', 
                color='Prediction', 
                tooltip=['Prediction','Counts']
            )
            st.altair_chart(pc, use_container_width=True)
            
            st.download_button(
                label="Download Predictions CSV",
                data=df_emotions.to_csv(index=False),
                file_name='emotion_predictions.csv',
                mime='text/csv'
            )
    
    else:
        add_page_visited_details("About", datetime.now())
        st.subheader("About")
        st.markdown("""
        **Emotion Classifier App**  
        - Built with Streamlit, scikit-learn, and Altair/Plotly for visualization.  
        - Tracks user interactions and predictions.  
        - Supports top-3 emotion predictions with confidence scores.  
        - Designed for educational/demo purposes.
        """)

if __name__ == '__main__':
    main()
