import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import joblib
import os

def train_collaborative_model(data_path='processed_data/reviews.csv', model_dir='models'):
    """
    Trains and saves an SVD collaborative filtering model.
    """
    print("Training collaborative filtering model...")
    reviews_df = pd.read_csv(data_path)
    
    # Surprise library requires data in a specific format
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(reviews_df[['user_id', 'tour_id', 'stars']], reader)
    
    # Split the data
    trainset, testset = train_test_split(data, test_size=0.2)
    
    # Use the SVD algorithm
    model = SVD()
    
    # Train the model
    model.fit(trainset)
    
    # Save the trained model
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'collaborative_svd_model.pkl'))
    
    print("Collaborative filtering model trained and saved.")
    return model

if __name__ == '__main__':
    # Example of how to train the model
    train_collaborative_model(
        data_path='data/processed_data/reviews.csv', 
        model_dir='models'
    )
    print("\nTo get recommendations, load this model in your main application.")
