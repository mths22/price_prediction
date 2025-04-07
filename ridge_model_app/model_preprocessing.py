import pickle
import pandas as pd
import numpy as np
import re
import string
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import StandardScaler

# Define the custom tokenizer
forbidden_char = string.punctuation + "1234567890"

def custom_tokenizer(text):
    text = text.lower()
    text = text.replace("/", " ")
    text = re.sub(f"[{forbidden_char}]", "", text)
    tokens = text.split()
    tokens = [tok for tok in tokens if tok not in ENGLISH_STOP_WORDS and len(tok) > 2]
    return tokens

# Load your saved vectorizers
def load_vectorizers():
    with open('saved_models/vectorizer_name.pkl', 'rb') as f:
        vectorizer_name = pickle.load(f)

    with open('saved_models/vectorizer_desc.pkl', 'rb') as f:
        vectorizer_desc = pickle.load(f)

    with open('saved_models/vectorizer_brand.pkl', 'rb') as f:
        vectorizer_brand = pickle.load(f)

    with open('saved_models/vectorizer_cat.pkl', 'rb') as f:
        vectorizer_cat = pickle.load(f)
    
    return vectorizer_name, vectorizer_desc, vectorizer_brand, vectorizer_cat


# Preprocessing function for new data
def preprocess_data(df):
    vectorizer_name, vectorizer_desc, vectorizer_brand, vectorizer_cat = load_vectorizers()

    # Ensure all required columns are present
    required_columns = ['name', 'item_description', 'brand_name', 'category_name', 'shipping', 'item_condition_id']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"The following required columns are missing: {missing_columns}")

    # Apply vectorizers to text data
    name_features = vectorizer_name.transform(df['name'].astype(str))
    desc_features = vectorizer_desc.transform(df['item_description'].astype(str))
    brand_features = vectorizer_brand.transform(df['brand_name'].astype(str))
    cat_features = vectorizer_cat.transform(df['category_name'].astype(str))

    # Combine text features
    text_features = hstack([name_features, desc_features, brand_features, cat_features])

    # Include numeric columns 'shipping' and 'item_condition_id'
    numeric_features = df[['shipping', 'item_condition_id']].to_numpy()  # Ensure these are numeric
    from scipy.sparse import csr_matrix
    numeric_features_sparse = csr_matrix(numeric_features)

    # Combine text and numeric features
    combined_features = hstack([text_features, numeric_features_sparse])
    
    return combined_features


# Load your trained Ridge model
def load_model():
    with open('saved_models/ridge_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


