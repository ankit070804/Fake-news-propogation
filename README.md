🕵️ Fake News Propagation Intelligence System
📌 Overview

This project is an AI-powered system that detects whether a news headline is fake or real and analyzes how it spreads across a network over time.
It combines Machine Learning, simulation modeling, and interactive visualization to provide insights into the lifecycle of news propagation.

🚀 Features
🧠 Fake News Detection using TF-IDF + Machine Learning.
📊 Propagation Simulation based on SIR (Susceptible-Infected-Recovered) model

🌐 Network Graph Visualization using NetworkX

📈 Interactive Charts using Plotly
⚡ Real-time Dashboard built with Streamlit

📄 Automated Report Generation

⚠️ Risk Scoring System for news credibility


🛠️ Tech Stack

Programming: Python

Machine Learning: Scikit-learn

Data Processing: Pandas, NumPy

Visualization: Plotly, NetworkX

Web App: Streamlit


📊 How It Works

User inputs a news headline and parameters (tweets, virality, etc.)

Model predicts probability of fake vs real news

System simulates how the news spreads over time


Graphs show:

Spread (people sharing)

Recovery (people stopping belief)

Network graph represents how information flows


▶️ Run Locally

pip install -r requirements.txt

streamlit run fake_news.py


📁 Project Structure

├──Fake_news.ipynb

├──politifact_fake.csv && politifact_real.csv

├── fake_news.py

├── fake_news_model.pkl

├── tfidf_vectorizer.pkl

├── requirements.txt

└── README.md


🎯 Use Cases

Fake news detection and analysis

Social media monitoring

Research on information spread

Educational demonstration of SIR model


📈 Future Improvements

Integration with real-time social media APIs

Advanced NLP models (BERT, Transformers)
Improved accuracy with larger datasets
Deployment on cloud platforms
