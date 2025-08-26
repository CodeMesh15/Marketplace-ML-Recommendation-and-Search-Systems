import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib
import os
from feature_engineering import create_ranking_features, generate_user_features

def train_ranking_model(processed_data_dir='processed_data', model_dir='models'):
    """
    Trains and saves a LightGBM ranking model.
    """
    print("Training ranking model...")
    
    # Load data
    reviews_df = pd.read_csv(os.path.join(processed_data_dir, 'reviews.csv'))
    tours_df = pd.read_csv(os.path.join(processed_data_dir, 'tours.csv'))
    
    # Generate user features
    users_df = generate_user_features(reviews_df)
    
    # Our training data is the reviews dataframe, which contains (user, tour) pairs
    # The target variable is the 'stars' rating the user gave.
    training_pairs = reviews_df[['user_id', 'tour_id', 'stars']]
    
    # Create features for these pairs
    features_df = create_ranking_features(training_pairs, tours_df, users_df)
    
    # Define features (X) and target (y)
    X = features_df.drop(columns=['user_id', 'tour_id', 'stars', 'name', 'city', 'state', 'categories'])
    y = features_df['stars']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train LightGBM Regressor
    # We use regression to predict the star rating, which acts as a relevance score.
    ranker = lgb.LGBMRegressor(objective='regression_l1', random_state=42)
    ranker.fit(X_train, y_train,
               eval_set=[(X_test, y_test)],
               eval_metric='rmse',
               callbacks=[lgb.early_stopping(10)])

    # Save the trained model
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(ranker, os.path.join(model_dir, 'ranking_model.pkl'))
    
    print("Ranking model trained and saved.")

if __name__ == '__main__':
    train_ranking_model(
        processed_data_dir='data/processed_data',
        model_dir='models'
    )
