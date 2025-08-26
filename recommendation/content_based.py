import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

def build_content_model(data_path='processed_data/tours.csv', model_dir='models'):
    """
    Builds and saves a TF-IDF vectorizer and a cosine similarity matrix.
    """
    print("Building content-based model...")
    tours_df = pd.read_csv(data_path)
    
    # Combine text features into a single 'content' string
    tours_df['content'] = tours_df['name'] + ' ' + tours_df['categories'] + ' ' + tours_df['city']
    
    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(tours_df['content'])
    
    # Calculate cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Save the model components
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(cosine_sim, os.path.join(model_dir, 'content_similarity_matrix.pkl'))
    joblib.dump(tours_df, os.path.join(model_dir, 'content_tours_df.pkl'))
    
    print("Content-based model built and saved successfully.")
    return cosine_sim, tours_df

def get_content_recommendations(tour_id, cosine_sim, tours_df, top_n=10):
    """
    Gets top N similar tours for a given tour_id.
    """
    # Get the index of the tour that matches the ID
    idx = tours_df.index[tours_df['tour_id'] == tour_id].tolist()[0]
    
    # Get the pairwise similarity scores of all tours with that tour
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the tours based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the top_n most similar tours (excluding the tour itself)
    sim_scores = sim_scores[1:top_n+1]
    
    # Get the tour indices
    tour_indices = [i[0] for i in sim_scores]
    
    # Return the top_n most similar tour IDs
    return tours_df['tour_id'].iloc[tour_indices].tolist()

if __name__ == '__main__':
    # Example of how to build the model and get recommendations
    sim_matrix, df = build_content_model(
        data_path='data/processed_data/tours.csv', 
        model_dir='models'
    )
    # Example: Find tours similar to the first tour in the dataframe
    example_tour_id = df['tour_id'].iloc[0]
    recommendations = get_content_recommendations(example_tour_id, sim_matrix, df)
    print(f"\nRecommendations for tour '{df.iloc[0]['name']}':")
    print(recommendations)
