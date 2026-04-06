🕵️ Fake News Propagation Intelligence System

The Fake News Propagation Intelligence System is an AI-powered application that detects whether a news headline is fake or real and analyzes how it spreads over time. The system uses machine learning techniques such as TF-IDF vectorization and classification models to predict the credibility of news content. It further simulates the spread of information using an SIR (Susceptible-Infected-Recovered) model, treating news propagation similar to a viral process. The application provides interactive visualizations using Plotly and NetworkX to display propagation trends and network structures. Built with Streamlit, the system offers a real-time dashboard where users can input parameters like tweet volume and virality to observe spread dynamics, risk levels, and generate reports. This project demonstrates the integration of machine learning, data visualization, and simulation modeling to understand and analyze the impact of fake news.


📁 Project Structure

├──Fake_news.ipynb

├──politifact_fake.csv && politifact_real.csv

├── fake_news.py

├── fake_news_model.pkl

├── tfidf_vectorizer.pkl

├── requirements.txt

└── README.md
