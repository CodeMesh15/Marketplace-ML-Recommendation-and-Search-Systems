import pandas as pd
from rank_bm25 import BM25Okapi
import joblib
import os

def build_search_index(data_path='processed_data/tours.csv', model_dir='models'):
    """
    Creates and saves a BM25 search index.
    """
    print("Building search index...")
    tours_df = pd.read_csv(data_path)
    
    # Create a text 'document' for each tour
    tours_df['document'] = tours_df['name'] + ' ' + tours_df['categories'] + ' ' + tours_df['city']
    
    # Tokenize the documents (simple split)
    tokenized_corpus = [doc.lower().split() for doc in tours_df['document'].fillna('')]
    
    # Create the BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save the index and the dataframe for mapping
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(bm25, os.path.join(model_dir, 'search_bm25_index.pkl'))
    joblib.dump(tours_df, os.path.join(model_dir, 'search_tours_df.pkl'))
    
    print("Search index built and saved.")

if __name__ == '__main__':
    build_search_index(
        data_path='data/processed_data/tours.csv',
        model_dir='models'
    )
