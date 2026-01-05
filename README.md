# 🎵 Spotify Lyric Search Engine

A Machine Learning project that identifies song titles and artists based on input lyric snippets. This project uses **TensorFlow** and the **Universal Sentence Encoder** to perform semantic text similarity search, allowing users to find songs even if they don't remember the exact words.

## 🚀 Overview

* **Task:** Identify Song Title and Artist from a text snippet.
* **Dataset:** Spotify Million Song Dataset (~57,000 songs).
* **Method:** Semantic Search using Deep Learning embeddings.
* **Model:** Universal Sentence Encoder (via TensorFlow Hub).

## 🛠️ Tech Stack

* **Language:** Python 3.8+
* **Libraries:**
    * `pandas` (Data Manipulation)
    * `tensorflow` & `tensorflow_hub` (Deep Learning)
    * `scikit-learn` (Cosine Similarity)
    * `numpy` (Math operations)

## 📂 Project Structure

```text
Spotify-Lyric-Search/
├── data/
│   └── spotify_millsongdata.csv  <-- (Download this from Kaggle)
├── lyric_search.py               # Main Python script
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
