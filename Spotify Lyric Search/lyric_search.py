import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import re

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
def load_and_clean_data(filepath):
    print("Loading dataset...")
    # Loading the dataset (Assumes 'text' column contains lyrics)
    # Common Kaggle Dataset: spotify_millsongdata.csv
    df = pd.read_csv(filepath)
    
    # We will sample 5000 songs for speed in this demo. 
    # Comment out .sample() to run on the full 57k dataset (requires more RAM).
    df = df.sample(n=5000, random_state=42).reset_index(drop=True)
    
    print(f"Data loaded: {len(df)} songs.")
    
    # Preprocessing function
    def preprocess_text(text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  # Remove newlines/extra spaces
        text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
        return text.strip()

    print("Preprocessing lyrics...")
    df['clean_text'] = df['text'].apply(preprocess_text)
    return df

# ==========================================
# 2. MODEL SETUP (TensorFlow / TF Hub)
# ==========================================
def build_model():
    print("Loading Universal Sentence Encoder (this may take a minute)...")
    # This downloads a pre-trained model capable of understanding sentence semantics
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    return model

# ==========================================
# 3. GENERATE EMBEDDINGS
# ==========================================
def vectorize_database(model, df, batch_size=100):
    print("Vectorizing lyrics database...")
    # We process in batches to avoid OOM (Out Of Memory) errors
    embeddings = []
    
    for i in range(0, len(df), batch_size):
        batch = df['clean_text'][i : i+batch_size].tolist()
        batch_emb = model(batch)
        embeddings.append(batch_emb)
        if i % 1000 == 0:
            print(f"Processed {i}/{len(df)}")
            
    # Concatenate all batches into one large matrix
    embedding_matrix = tf.concat(embeddings, axis=0)
    return embedding_matrix

# ==========================================
# 4. SEARCH ALGORITHM
# ==========================================
def search_lyrics(query, model, embedding_matrix, df, top_k=3):
    # 1. Preprocess query
    query = query.lower()
    
    # 2. Vectorize query
    query_vec = model([query])
    
    # 3. Calculate Cosine Similarity between query and ALL songs
    # Result is a similarity score for every song in the DB
    similarities = cosine_similarity(query_vec, embedding_matrix)
    
    # 4. Get top K indices
    # flatten() converts structure [[0.1, 0.9...]] to [0.1, 0.9...]
    top_indices = similarities[0].argsort()[-top_k:][::-1]
    
    print(f"\n--- Search Results for: '{query}' ---")
    results = []
    for idx in top_indices:
        score = similarities[0][idx]
        song = df.iloc[idx]
        print(f"Confidence: {score:.4f} | Song: {song['song']} | Artist: {song['artist']}")
        results.append((song['song'], song['artist'], score))
    
    return results

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Path to your dataset
    # Ensure you have 'spotify_millsongdata.csv' in the folder
    dataset_path = 'spotify_millsongdata.csv' 
    
    try:
        # 1. Load Data
        df = load_and_clean_data(dataset_path)
        
        # 2. Load Model
        model = build_model()
        
        # 3. Vectorize Data (Run once and keep in memory)
        embedding_matrix = vectorize_database(model, df)
        
        # 4. Test the Search
        while True:
            user_input = input("\nEnter lyric snippet (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            search_lyrics(user_input, model, embedding_matrix, df)
            
    except FileNotFoundError:
        print(f"Error: '{dataset_path}' not found. Please download the dataset from Kaggle.")