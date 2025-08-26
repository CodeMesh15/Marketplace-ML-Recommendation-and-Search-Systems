import pandas as pd

def create_ranking_features(pairs_df, tours_df, users_df):
    """
    Creates features for a dataframe of (user, tour) pairs.
    
    Args:
        pairs_df (pd.DataFrame): DataFrame with 'user_id' and 'tour_id' columns.
        tours_df (pd.DataFrame): DataFrame with tour features.
        users_df (pd.DataFrame): DataFrame with user features.
        
    Returns:
        pd.DataFrame: A new DataFrame with engineered features.
    """
    # Merge tour features
    features_df = pd.merge(pairs_df, tours_df, on='tour_id', how='left')
    
    # Merge user features
    features_df = pd.merge(features_df, users_df, on='user_id', how='left')
    
    # Fill any missing values that might result from merges
    features_df.fillna(0, inplace=True)
    
    return features_df

def generate_user_features(reviews_df):
    """
    Generates aggregate features for each user.
    """
    user_features = reviews_df.groupby('user_id')['stars'].agg(['mean', 'count']).reset_index()
    user_features.columns = ['user_id', 'user_avg_rating', 'user_review_count']
    return user_features
