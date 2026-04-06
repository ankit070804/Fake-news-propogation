import streamlit as st
import pickle, random
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from scipy.sparse import hstack
from datetime import datetime

# ─────────────────── CONFIG ───────────────────
st.set_page_config("Fake News Intelligence", "🕵️", layout="wide")

# ─────────────────── LOAD MODEL ───────────────────
@st.cache_resource
def load_model():
    try:
        return (
            pickle.load(open("fake_news_model.pkl","rb")),
            pickle.load(open("tfidf_vectorizer.pkl","rb")),
            True
        )
    except:
        return None, None, False

model, tfidf, model_loaded = load_model()

# ─────────────────── CORE FUNCTIONS ───────────────────
def simulate(num_tweets, is_fake, is_viral, density, hours):
    spread = (0.08 if is_fake else 0.04) * (2.5 if is_viral else 1) * (1 + density/500)
    recovery = 0.03 if is_fake else 0.07

    pop = max(num_tweets*100, 5000)
    inf, sus, rec = max(num_tweets,10), pop-max(num_tweets,10), 0

    data=[]
    for h in range(hours+1):
        new_i = min(sus, int(spread*inf*(sus/pop)))
        new_r = int(recovery*inf)
        sus -= new_i; inf += new_i-new_r; rec += new_r
        data.append([h, inf, rec, pop-sus])
    return pd.DataFrame(data, columns=["hour","spread","recovered","reach"])

def build_graph(n, is_fake):
    G = nx.barabasi_albert_graph(n,2)
    nx.set_node_attributes(G, {i:random.random() for i in G.nodes()}, "belief")
    return G

def predict(title, num_tweets, density, viral, cat):
    if model_loaded:
        X = hstack([
            tfidf.transform([title]),
            np.array([[num_tweets,np.log(num_tweets+1),density,viral,cat]])
        ])
        pred = model.predict(X)[0]
        try:
            p = model.predict_proba(X)[0]
            return pred, p[0], p[1]
        except:
            return pred, 0.7, 0.3
    else:
        score = sum(w in title.lower() for w in ["breaking","shocking","secret"])
        fake_prob = min(0.9,0.4+score*0.2+viral*0.1)
        return (0 if fake_prob>0.5 else 1), fake_prob, 1-fake_prob

# ─────────────────── SIDEBAR ───────────────────
with st.sidebar:
    title = st.text_input("Headline")
    tweets = st.slider("Tweets",0,5000,300)
    density = st.slider("Density",0.0,500.0,50.0)
    viral = st.selectbox("Viral",[0,1])
    cat = st.selectbox("Spread",[0,1])
    run = st.button("Analyze")

# ─────────────────── MAIN ───────────────────
if run and title:
    pred, fake_p, real_p = predict(title, tweets, density, viral, cat)
    is_fake = pred == 0

    st.title("📊 Fake News Analysis")

    # ─── RESULT ───
    st.subheader("Result")
    st.write("Fake Prob:", round(fake_p,2))
    st.write("Real Prob:", round(real_p,2))
    st.success("FAKE" if is_fake else "REAL")

    # ─── SIMULATION ───
    df = simulate(tweets, is_fake, viral, density, 48)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.hour,y=df.spread,name="Spread"))
    fig.add_trace(go.Scatter(x=df.hour,y=df.recovered,name="Recovered"))
    st.plotly_chart(fig,use_container_width=True)

    # ─── NETWORK ───
    G = build_graph(50, is_fake)
    st.write("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

    # ─── REPORT ───
    report = f"""
    Headline: {title}
    Fake: {fake_p:.2f}
    Real: {real_p:.2f}
    """
    st.download_button("Download Report", report)