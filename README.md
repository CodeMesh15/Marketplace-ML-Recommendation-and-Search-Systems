# Marketplace-ML: Recommendation and Search Systems

An implementation of the core machine learning systems for a tours and activities marketplace. This project covers recommendation systems (content-based and collaborative), feed ranking, and search retrieval.

---

## 1. Project Overview

This project aims to build the foundational AI systems for a two-sided marketplace connecting users with local tours. As the founding AI lead, the work involves creating systems from the ground up to enhance user experience and engagement. The project is divided into three core components: a recommendation engine, a feed ranking model, and a search retrieval system.

---

## 2. Core Objectives

-   To build a **recommendation system** that suggests relevant tours and activities to users.
-   To implement a **feed ranking model** to personalize the order of items shown to a user.
-   To create a **search retrieval system** that returns the most relevant tours for a given user query.

---

## 3. Methodology

This project is broken down into three independent but related modules.

### Module 1: Recommendation System

1.  **Dataset**: We'll use a public dataset that mimics a user-item interaction scenario, such as the [Yelp Dataset](https://www.yelp.com/dataset) (filtering for "Tours" or "Active Life" categories) or the [Airbnb Listings Dataset](http://insideairbnb.com/get-the-data/). The data should contain users, items (tours), and interactions (reviews/bookings).
2.  **Approach 1: Content-Based Filtering**:
    -   **Feature Extraction**: Convert tour descriptions and attributes (e.g., location, category, price) into numerical vectors using **TF-IDF**.
    -   **Similarity Calculation**: Use **Cosine Similarity** to find tours similar to the ones a user has previously liked or booked.
3.  **Approach 2: Collaborative Filtering**:
    -   **User-Item Matrix**: Create a sparse matrix of user interactions with tours.
    -   **Matrix Factorization**: Use algorithms like **Singular Value Decomposition (SVD)** or **Alternating Least Squares (ALS)** to learn latent embeddings for users and items. The dot product of these embeddings predicts user ratings for unseen items.

### Module 2: Feed Ranking

1.  **Problem Formulation**: This is a learning-to-rank problem. Given a list of candidate tours (perhaps from the recommendation system), we want to re-order them in the most engaging way for a specific user.
2.  **Feature Engineering**: For each (user, tour) pair, create features like:
    -   **User Features**: User's past booking frequency, preferred tour categories.
    -   **Item Features**: Tour's popularity, average rating, price.
    -   **Interaction Features**: Cosine similarity between user and item embeddings (from Module 1).
3.  **Model**: Train a gradient boosting model (like **LightGBM** or **XGBoost**) to predict a relevance score (e.g., probability of a click or booking). The feed for a user is then sorted by this score.

### Module 3: Search Retrieval System

1.  **Problem Formulation**: This is an information retrieval task. Given a text query (e.g., "walking tours in Paris"), we need to return a ranked list of relevant tours.
2.  **Indexing**:
    -   Create a "document" for each tour containing its title, description, location, and other text fields.
    -   Use a library like **Elasticsearch** or build a simpler version using **TF-IDF** or **BM25** (Best Matching 25) to index these documents.
3.  **Retrieval**:
    -   When a user enters a query, convert the query to a vector.
    -   Calculate the similarity score (e.g., BM25 score) between the query vector and all indexed tour documents.
    -   Return the tours with the highest scores.
