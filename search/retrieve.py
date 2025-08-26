import joblib

def search_tours(query, bm25_index, tours_df, top_n=10):
    """
    Searches for tours using a query against the BM25 index.
    
    Args:
        query (str): The user's search query.
        bm25_index: The loaded BM25Okapi model.
        tours_df (pd.DataFrame): The DataFrame of tours.
        top_n (int): The number of results to return.
        
    Returns:
        list: A list of top_n relevant tour IDs.
    """
    # Tokenize the query
    tokenized_query = query.lower().split()
    
    # Get document scores
    doc_scores = bm25_index.get_scores(tokenized_query)
    
    # Get the top N indices
    top_indices = doc_scores.argsort()[::-1][:top_n]
    
    # Get the corresponding tour IDs
    top_tour_ids = tours_df.iloc[top_indices]['tour_id'].tolist()
    
    return top_tour_ids

if __name__ == '__main__':
    # Example of how to load the index and perform a search
    index = joblib.load('models/search_bm25_index.pkl')
    df = joblib.load('models/search_tours_df.pkl')
    
    test_query = "art museums"
    results = search_tours(test_query, index, df)
    
    print(f"Search results for '{test_query}':")
    print(results)
