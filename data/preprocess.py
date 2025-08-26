
import pandas as pd
import argparse
import os
import json

def preprocess_data(input_dir, output_dir):
    """
    Loads raw Yelp data, filters for tour-related businesses,
    and saves cleaned data to CSV files.
    """
    print("Starting data preprocessing...")

    # Define paths to raw data
    business_path = os.path.join(input_dir, 'yelp_academic_dataset_business.json')
    review_path = os.path.join(input_dir, 'yelp_academic_dataset_review.json')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Process Businesses to find Tours ---
    print(f"Loading businesses from {business_path}...")
    
    # Load the business data line by line to handle potential JSON errors
    with open(business_path, 'r', encoding='utf-8') as f:
        businesses_data = [json.loads(line) for line in f]
    businesses_df = pd.DataFrame(businesses_data)

    # Filter for businesses that are likely tours or activities
    tour_categories = ['Tours', 'Active Life', 'Arts & Entertainment', 'Local Flavor']
    
    # Drop rows with missing categories
    businesses_df.dropna(subset=['categories'], inplace=True)
    
    # Find businesses that have at least one of our desired categories
    tours_df = businesses_df[businesses_df['categories'].str.contains('|'.join(tour_categories))]

    # Select and rename columns for clarity
    tours_df = tours_df[['business_id', 'name', 'city', 'state', 'stars', 'review_count', 'categories']]
    tours_df = tours_df.rename(columns={'business_id': 'tour_id'})

    # Save the cleaned tours data
    tours_output_path = os.path.join(output_dir, 'tours.csv')
    tours_df.to_csv(tours_output_path, index=False)
    print(f"Found {len(tours_df)} tours. Saved to {tours_output_path}")

    # --- 2. Process Reviews for the selected Tours ---
    print(f"Loading reviews from {review_path}...")
    
    # Get the set of tour IDs for efficient filtering
    valid_tour_ids = set(tours_df['tour_id'])
    
    # Load reviews in chunks as the file can be very large
    chunk_size = 100000
    review_chunks = pd.read_json(review_path, lines=True, chunksize=chunk_size)

    filtered_reviews = []
    for chunk in review_chunks:
        # Filter reviews that belong to our list of tours
        filtered_chunk = chunk[chunk['business_id'].isin(valid_tour_ids)]
        filtered_reviews.append(filtered_chunk)
    
    reviews_df = pd.concat(filtered_reviews, ignore_index=True)
    
    # Select and rename columns
    reviews_df = reviews_df[['review_id', 'user_id', 'business_id', 'stars', 'text', 'date']]
    reviews_df = reviews_df.rename(columns={'business_id': 'tour_id'})

    # Save the cleaned reviews data
    reviews_output_path = os.path.join(output_dir, 'reviews.csv')
    reviews_df.to_csv(reviews_output_path, index=False)
    print(f"Found {len(reviews_df)} reviews for the tours. Saved to {reviews_output_path}")
    
    print("Data preprocessing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess raw Yelp data for the Marketplace-ML project.")
    
    parser.add_argument('--input_dir', type=str, default='raw_data',
                        help='Directory containing the raw yelp .json files.')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                        help='Directory to save the cleaned .csv files.')
                        
    args = parser.parse_args()
    
    preprocess_data(args.input_dir, args.output_dir)

    # Example Usage from command line (after running get_data.sh):
    # python data/preprocess.py --input_dir data/raw_data --output_dir data/processed_data
