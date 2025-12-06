import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import errorcode
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
from datetime import datetime
import gc

# --- DATABASE CONNECTION CONFIGURATION ---
DB_CONFIG = {
    'user': 'root',
    'password': 'Bandit409023!',
    'host': 'localhost', 
    'port': '3305',
    'database': 'book_reviews'
}

# --- PERFORMANCE CONFIGURATION ---
BATCH_SIZE = 10000  # Process sentiment in batches
MAX_FEATURES = 3000  # Limit TF-IDF features for memory efficiency
MIN_DF = 5  # Ignore rare terms
MAX_DF = 0.7  # Ignore very common terms

def log_time(message):
    """Print message with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

# ---------------------------------------------
# STEP 1: Load Data with Batch Processing Support
# ---------------------------------------------
def load_data_from_db(limit=None):
    """
    Load data with optional row limit for testing.
    Set limit=10000 for quick testing, None for full dataset.
    """
    log_time("=" * 70)
    log_time("DATABASE CONNECTION")
    log_time("=" * 70)
    print(f"Host: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print(f"Database: {DB_CONFIG['database']}")
    print()
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        log_time("✓ Connected to database")
        
        # Build query with optional limit
        query = """
        SELECT reviewtext, reviewscore, title
        FROM reviews
        """
        if limit:
            query += f" LIMIT {limit}"
            log_time(f"Loading first {limit:,} reviews (test mode)...")
        else:
            log_time("Loading ALL reviews (this may take a few minutes)...")
        
        query += ";"
        
        # Load data in chunks for memory efficiency
        chunk_size = 50000
        chunks = []
        
        for chunk in pd.read_sql(query, conn, chunksize=chunk_size):
            chunks.append(chunk)
            log_time(f"  Loaded {len(chunk):,} rows... (Total: {sum(len(c) for c in chunks):,})")
        
        df = pd.concat(chunks, ignore_index=True)
        conn.close()
        
        log_time(f"✓ Loaded {len(df):,} reviews")
        log_time(f"✓ Unique books: {df['title'].nunique():,}")
        print()
        
        return df
    
    except mysql.connector.Error as err:
        print("\n" + "=" * 70)
        print("DATABASE ERROR")
        print("=" * 70)
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("✗ Access denied: Check username/password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print(f"✗ Database '{DB_CONFIG['database']}' does not exist")
        else:
            print(f"✗ Error: {err}")
        print("=" * 70)
        print()
        return pd.DataFrame()
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return pd.DataFrame()

# ---------------------------------------------
# STEP 2: Download NLTK Resources
# ---------------------------------------------
def setup_nltk():
    """Ensure NLTK VADER lexicon is available."""
    log_time("Checking NLTK resources...")
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
        log_time("✓ VADER lexicon found")
    except LookupError:
        log_time("Downloading VADER lexicon...")
        nltk.download('vader_lexicon')
        log_time("✓ VADER lexicon downloaded")
    print()

# ---------------------------------------------
# STEP 3: Process Reviews with Batching
# ---------------------------------------------
def process_reviews_batch(df):
    """Process reviews in batches to manage memory."""
    log_time("=" * 70)
    log_time("PROCESSING REVIEWS")
    log_time("=" * 70)
    
    # Clean data
    initial_count = len(df)
    df = df.dropna(subset=['reviewtext'])
    df = df.reset_index(drop=True)
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        log_time(f"✓ Removed {removed_count:,} reviews with missing text")
    
    log_time(f"✓ Processing {len(df):,} valid reviews")
    
    # Sentiment analysis in batches
    log_time(f"Calculating sentiment scores (batch size: {BATCH_SIZE:,})...")
    sia = SentimentIntensityAnalyzer()
    
    sentiments = []
    for i in range(0, len(df), BATCH_SIZE):
        batch = df['reviewtext'].iloc[i:i+BATCH_SIZE]
        batch_sentiments = batch.apply(lambda x: sia.polarity_scores(str(x))['compound'])
        sentiments.extend(batch_sentiments)
        
        if (i // BATCH_SIZE + 1) % 10 == 0:
            log_time(f"  Processed {i+len(batch):,}/{len(df):,} reviews...")
    
    df['sentiment'] = sentiments
    log_time("✓ Sentiment analysis complete")
    
    # Normalize ratings
    min_rating = df['reviewscore'].min()
    max_rating = df['reviewscore'].max()
    log_time(f"✓ Rating range: {min_rating} to {max_rating}")
    
    if max_rating > min_rating:
        df['rating_norm'] = (df['reviewscore'] - min_rating) / (max_rating - min_rating)
    else:
        df['rating_norm'] = 0.5
    
    # Combined score
    df['combined_score'] = 0.7 * df['sentiment'] + 0.3 * df['rating_norm']
    
    # Calculate book-level statistics
    log_time("Calculating book-level statistics...")
    book_stats = df.groupby('title').agg({
        'combined_score': 'mean',
        'reviewtext': 'count',
        'reviewscore': 'mean',
        'sentiment': 'mean'
    }).rename(columns={'reviewtext': 'review_count'})
    
    df = df.merge(book_stats.add_suffix('_book'), on='title', how='left')
    
    log_time("✓ Book statistics calculated")
    print()
    
    return df

# ---------------------------------------------
# STEP 4: Optimized Content Model (Memory Efficient)
# ---------------------------------------------
def build_optimized_content_model(df):
    """
    Build TF-IDF model WITHOUT creating full similarity matrix.
    This is the key optimization for large datasets.
    """
    log_time("=" * 70)
    log_time("BUILDING CONTENT MODEL (MEMORY OPTIMIZED)")
    log_time("=" * 70)
    log_time("Creating TF-IDF vectors...")
    
    # Ensure all text is strings
    df['reviewtext'] = df['reviewtext'].astype(str)
    
    # More aggressive feature limiting for large datasets
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=(1, 2),
        dtype=np.float32  # Use 32-bit floats instead of 64-bit
    )
    
    log_time(f"  max_features={MAX_FEATURES}, min_df={MIN_DF}, max_df={MAX_DF}")
    tfidf_matrix = tfidf.fit_transform(df['reviewtext'])
    
    log_time(f"✓ TF-IDF matrix shape: {tfidf_matrix.shape}")
    log_time(f"✓ Vocabulary size: {len(tfidf.vocabulary_):,}")
    log_time(f"✓ Matrix sparsity: {(1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100:.2f}%")
    log_time(f"✓ Approx memory: {tfidf_matrix.data.nbytes / (1024**2):.1f} MB")
    print()
    
    # DO NOT create full similarity matrix - will be computed on demand
    return tfidf, tfidf_matrix

# ---------------------------------------------
# STEP 5: On-Demand Recommendation (No Pre-computation)
# ---------------------------------------------
def recommend_books_optimized(df, tfidf, tfidf_matrix, selected_title, 
                               top_n=10, text_weight=0.6, score_weight=0.4):
    """
    Generate recommendations WITHOUT pre-computing full similarity matrix.
    This computes similarity only for the target book.
    """
    # Find matching book (case-insensitive)
    matching_titles = df[df['title'].str.lower() == selected_title.lower()]['title'].unique()
    
    if len(matching_titles) == 0:
        print(f"\n✗ Book '{selected_title}' not found")
        print("\nAvailable books (sample):")
        for i, title in enumerate(df['title'].unique()[:15], 1):
            reviews = len(df[df['title'] == title])
            print(f"  {i}. {title} ({reviews} reviews)")
        if df['title'].nunique() > 15:
            print(f"  ... and {df['title'].nunique() - 15:,} more books")
        return None
    
    actual_title = matching_titles[0]
    book_reviews = df[df['title'] == actual_title]
    
    log_time(f"Analyzing '{actual_title}' ({len(book_reviews)} reviews)...")
    
    # Get TF-IDF vectors for this book's reviews
    book_indices = book_reviews.index.tolist()
    target_vectors = tfidf_matrix[book_indices]
    
    # Compute average TF-IDF vector for this book
    # Convert to array to avoid np.matrix deprecation warning
    target_vector = np.asarray(target_vectors.mean(axis=0))
    
    # Compute similarity ONLY between target and all reviews
    # This is much more memory efficient than full matrix
    log_time("Computing similarities (this may take 1-2 minutes for 1.5M reviews)...")
    
    # Process in batches to avoid memory issues
    batch_size = 50000
    all_similarities = []
    
    for i in range(0, tfidf_matrix.shape[0], batch_size):
        batch = tfidf_matrix[i:i+batch_size]
        batch_sim = cosine_similarity(target_vector, batch).flatten()
        all_similarities.extend(batch_sim)
        
        if (i // batch_size + 1) % 10 == 0:
            log_time(f"  Processed {i+len(batch_sim):,}/{tfidf_matrix.shape[0]:,} similarities...")
    
    similarities = np.array(all_similarities)
    log_time("✓ Similarity computation complete")
    
    # Normalize similarities
    if similarities.max() > similarities.min():
        similarities_norm = (similarities - similarities.min()) / (similarities.max() - similarities.min())
    else:
        similarities_norm = similarities
    
    # Create hybrid score
    hybrid_scores = (
        text_weight * similarities_norm +
        score_weight * df['combined_score_book'].values
    )
    
    # Add to dataframe
    df['hybrid_score'] = hybrid_scores
    
    # Group by book and average
    results = df.groupby('title').agg({
        'hybrid_score': 'mean',
        'review_count_book': 'first',
        'reviewscore_book': 'first',
        'sentiment_book': 'first'
    }).sort_values('hybrid_score', ascending=False)
    
    # Remove target book
    results = results[results.index != actual_title]
    
    # Get top N
    top_recommendations = results.head(top_n)
    
    # Display results
    print(f"\n{'='*70}")
    print(f"TOP {top_n} RECOMMENDATIONS FOR '{actual_title}'")
    print(f"{'='*70}")
    print(f"Algorithm: {text_weight*100:.0f}% Text Similarity + {score_weight*100:.0f}% Ratings/Sentiment")
    print()
    
    for i, (book, row) in enumerate(top_recommendations.iterrows(), 1):
        print(f"{i}. {book}")
        print(f"   Hybrid Score: {row['hybrid_score']:.4f}")
        print(f"   Reviews: {int(row['review_count_book']):,} | "
              f"Avg Rating: {row['reviewscore_book']:.2f} | "
              f"Sentiment: {row['sentiment_book']:+.3f}")
        print()
    
    # Clean up
    df.drop(columns=['hybrid_score'], inplace=True, errors='ignore')
    
    return top_recommendations

# ---------------------------------------------
# STEP 6: Dataset Summary
# ---------------------------------------------
def show_dataset_summary(df):
    """Display summary statistics."""
    log_time("=" * 70)
    log_time("DATASET SUMMARY")
    log_time("=" * 70)
    print(f"Total reviews: {len(df):,}")
    print(f"Unique books: {df['title'].nunique():,}")
    print(f"Average reviews per book: {len(df) / df['title'].nunique():.1f}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
    print()
    
    print("Rating distribution:")
    print(df['reviewscore'].value_counts().sort_index())
    print()
    
    print("Books with most reviews (top 10):")
    top_books = df.groupby('title').size().sort_values(ascending=False).head(10)
    for book, count in top_books.items():
        print(f"  {book}: {count:,} reviews")
    print()

# ---------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------
def main(test_mode=False):
    """
    Main execution function.
    
    Parameters:
    - test_mode: If True, loads only first 10,000 reviews for testing
    """
    print("\n" + "=" * 70)
    print("SCALABLE BOOK RECOMMENDATION SYSTEM")
    print("Optimized for Large Datasets (1M+ reviews)")
    print("=" * 70)
    print()
    
    # Step 1: Setup
    setup_nltk()
    
    # Step 2: Load data
    if test_mode:
        log_time("⚠️  RUNNING IN TEST MODE (10,000 reviews)")
        df = load_data_from_db(limit=10000)
    else:
        log_time("Running in FULL MODE (all reviews)")
        df = load_data_from_db(limit=None)
    
    if df.empty:
        log_time("✗ Cannot proceed without data")
        sys.exit(1)
    
    # Step 3: Process reviews
    df = process_reviews_batch(df)
    
    # Step 4: Build content model (no full similarity matrix)
    tfidf, tfidf_matrix = build_optimized_content_model(df)
    
    # Step 5: Show summary
    show_dataset_summary(df)
    
    # Step 6: Test recommendation
    log_time("=" * 70)
    log_time("TESTING RECOMMENDATIONS")
    log_time("=" * 70)
    
    test_title = df['title'].iloc[0]
    log_time(f"Using first book: '{test_title}'")
    print()
    
    recommendations = recommend_books_optimized(
        df, tfidf, tfidf_matrix, test_title, 
        top_n=10, text_weight=0.6, score_weight=0.4
    )
    
    if recommendations is not None:
        log_time("\n" + "=" * 70)
        log_time("✓ SYSTEM WORKING SUCCESSFULLY!")
        log_time("=" * 70)
        print("\nTo get recommendations for any book:")
        print("recommend_books_optimized(df, tfidf, tfidf_matrix, 'Book Title', top_n=10)")
    
    return df, tfidf, tfidf_matrix

# ---------------------------------------------
# RUN THE SYSTEM
# ---------------------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', 
                       help='Run in test mode (10K reviews only)')
    args = parser.parse_args()
    
    try:
        start_time = datetime.now()
        log_time(f"Started at {start_time.strftime('%H:%M:%S')}")
        print()
        
        df, tfidf, tfidf_matrix = main(test_mode=args.test)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        log_time(f"\nCompleted in {duration/60:.1f} minutes ({duration:.0f} seconds)")
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()