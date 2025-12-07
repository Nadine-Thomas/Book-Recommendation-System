import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re
from sqlalchemy import create_engine, text
import pickle
import os


class BookRecommendationSystem:
    def __init__(self, host='localhost', user='root', password='', database='book_reviews', port=3306):
        """Initialize database connection and load data"""
        # Create SQLAlchemy engine
        connection_string = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
        self.engine = create_engine(connection_string)
        self.books_df = None
        self.reviews_df = None
        self.book_profiles = None
        self.similarity_matrix = None
        self.tfidf_vectorizer = None
        self.cache_file = 'book_recommender_cache.pkl'

    def save_cache(self):
        """Save processed data to cache file"""
        print(f"\nSaving processed data to {self.cache_file}...")
        cache_data = {
            'books_df': self.books_df,
            'reviews_df': self.reviews_df,
            'book_profiles': self.book_profiles,
            'similarity_matrix': self.similarity_matrix,
            'tfidf_vectorizer': self.tfidf_vectorizer
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print("✓ Cache saved successfully!")

    def load_cache(self):
        """Load processed data from cache file"""
        if not os.path.exists(self.cache_file):
            return False

        print(f"Loading cached data from {self.cache_file}...")
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            self.books_df = cache_data['books_df']
            self.reviews_df = cache_data['reviews_df']
            self.book_profiles = cache_data['book_profiles']
            self.similarity_matrix = cache_data['similarity_matrix']
            self.tfidf_vectorizer = cache_data['tfidf_vectorizer']

            print(f"✓ Loaded {len(self.books_df)} books and {len(self.reviews_df)} reviews from cache")
            return True
        except Exception as e:
            print(f"✗ Error loading cache: {e}")
            return False

    def load_data(self, limit=10000, min_reviews=5):
        """Load books and reviews from database"""
        print(f"Loading up to {limit} books with at least {min_reviews} reviews...")

        # Load books that have minimum number of reviews, ordered by review count
        books_query = f"""
        SELECT b.bookid, b.title, b.description, b.authors, b.publisher, b.categories,
               COUNT(r.reviewid) as review_count,
               AVG(r.reviewscore) as avg_rating
        FROM books b
        LEFT JOIN reviews r ON b.bookid = r.bookid
        GROUP BY b.bookid
        HAVING review_count >= {min_reviews}
        ORDER BY review_count DESC
        LIMIT {limit}
        """
        self.books_df = pd.read_sql(books_query, self.engine)

        # Get bookids for these books
        book_ids = tuple(self.books_df['bookid'].tolist())

        # Load reviews for these books
        if len(book_ids) == 1:
            reviews_query = f"""
            SELECT reviewid, bookid, reviewscore, reviewsummary, reviewtext
            FROM reviews
            WHERE bookid = {book_ids[0]}
            """
        else:
            reviews_query = f"""
            SELECT reviewid, bookid, reviewscore, reviewsummary, reviewtext
            FROM reviews
            WHERE bookid IN {book_ids}
            """

        self.reviews_df = pd.read_sql(reviews_query, self.engine)

        print(f"Loaded {len(self.books_df)} books and {len(self.reviews_df)} reviews")
        print(f"Average reviews per book: {len(self.reviews_df) / len(self.books_df):.1f}")

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""

        # Convert to lowercase and remove special characters
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def remove_common_names(self, text):
        """Remove common character and author names that don't help with recommendations"""
        # Common first names that are often character names
        first_names = [
            'winston', 'julia', 'john', 'jane', 'mary', 'james', 'robert', 'michael',
            'william', 'david', 'richard', 'joseph', 'thomas', 'charles', 'daniel',
            'matthew', 'anthony', 'mark', 'donald', 'steven', 'paul', 'andrew', 'joshua',
            'harry', 'frodo', 'sam', 'ron', 'hermione', 'alice', 'elizabeth', 'darcy',
            'emma', 'scout', 'atticus', 'holden'
        ]

        # Split into words and filter
        words = text.split()
        filtered_words = [w for w in words if w not in first_names]

        return ' '.join(filtered_words)

    def create_book_profiles(self, weight_positive_reviews=True, remove_names=True):
        """Create text profiles for each book by combining all reviews"""
        print("Creating book profiles from reviews...")

        # Group reviews by book
        book_reviews = defaultdict(list)

        for _, review in self.reviews_df.iterrows():
            book_id = review['bookid']
            review_text = self.preprocess_text(review['reviewtext'])
            review_summary = self.preprocess_text(review['reviewsummary'])

            # Remove character names if requested
            if remove_names:
                review_text = self.remove_common_names(review_text)
                review_summary = self.remove_common_names(review_summary)

            # Combine review text and summary
            combined_text = f"{review_summary} {review_text}"

            # Weight positive reviews more (4-5 stars get repeated)
            if weight_positive_reviews and review['reviewscore'] >= 4:
                combined_text = combined_text + " " + combined_text

            if combined_text.strip():
                book_reviews[book_id].append(combined_text)

        # Create profiles
        profiles = []
        for _, book in self.books_df.iterrows():
            book_id = book['bookid']

            # Combine all reviews for this book
            reviews_text = ' '.join(book_reviews.get(book_id, []))

            # Add book metadata (categories and description weighted more)
            categories = self.preprocess_text(book['categories'])
            description = self.preprocess_text(book['description'])

            # Remove names from description too
            if remove_names:
                description = self.remove_common_names(description)

            # Weight categories and description heavily
            # Description often contains genre/theme keywords
            profile = f"{reviews_text} {categories} {categories} {categories} {categories} {description} {description}"

            profiles.append({
                'bookid': book_id,
                'title': book['title'],
                'profile': profile,
                'categories': book['categories'],
                'review_count': book['review_count']
            })

        self.book_profiles = pd.DataFrame(profiles)
        print(f"Created profiles for {len(self.book_profiles)} books")

    def calculate_similarity(self):
        """Calculate similarity between books using TF-IDF and cosine similarity"""
        print("Calculating book similarities...")

        # Enhanced stop words - add author names and common character names
        custom_stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words())

        # Add common author last names
        author_names = [
            'orwell', 'huxley', 'bradbury', 'rowling', 'tolkien', 'king',
            'christie', 'austen', 'dickens', 'shakespeare', 'hemingway',
            'fitzgerald', 'steinbeck', 'twain', 'poe', 'wilde', 'kafka',
            'vonnegut', 'salinger', 'collins', 'suzanne', 'stephenie', 'meyer'
        ]

        # Add book title references
        book_titles = [
            'gatsby', 'mockingbird', 'catcher', 'rye', 'hobbit', 'rings',
            'potter', 'games', 'twilight'
        ]

        custom_stop_words.extend(author_names)
        custom_stop_words.extend(book_titles)

        # Use TF-IDF to find important keywords
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=8000,
            min_df=2,
            max_df=0.6,
            ngram_range=(1, 3),
            stop_words=custom_stop_words,
            sublinear_tf=True
        )

        # Create TF-IDF matrix
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.book_profiles['profile']
        )

        # Calculate cosine similarity
        self.similarity_matrix = cosine_similarity(tfidf_matrix)

        print("Similarity calculation complete")

    def normalize_title(self, title):
        """Normalize title for duplicate detection"""
        if pd.isna(title):
            return ""
        # Remove common variations, punctuation, and extra whitespace
        normalized = str(title).lower()
        normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove punctuation
        normalized = re.sub(r'\s+', ' ', normalized).strip()  # Normalize whitespace
        # Remove common prefixes/suffixes
        normalized = re.sub(r'\b(the|a|an)\b', '', normalized)
        normalized = normalized.strip()
        return normalized

    def get_recommendations(self, book_title, n_recommendations=10, min_similarity=0.1,
                            same_category_boost=0.2, min_rec_reviews=20):
        """Get book recommendations based on a book title (diversity mode always enabled)"""

        # Find the book
        book_match = self.book_profiles[
            self.book_profiles['title'].str.contains(book_title, case=False, na=False, regex=False)
        ]

        if book_match.empty:
            print(f"Book '{book_title}' not found in database")
            return []

        # Get the index of the book
        book_idx = book_match.index[0]
        book_id = book_match.iloc[0]['bookid']
        book_category = book_match.iloc[0]['categories']
        source_title = book_match.iloc[0]['title']

        # Get the author of the source book
        source_author = self.books_df[self.books_df['bookid'] == book_id]['authors'].iloc[0]

        # Normalize source title for duplicate detection
        source_title_normalized = self.normalize_title(source_title)

        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[book_idx]))

        # Track seen titles to avoid duplicates
        seen_titles = set()
        seen_titles.add(source_title_normalized)

        # Boost scores for books in the same category
        boosted_scores = []
        for idx, score in sim_scores:
            if idx != book_idx:  # Don't include the book itself
                rec_title = self.book_profiles.iloc[idx]['title']
                rec_category = self.book_profiles.iloc[idx]['categories']
                rec_review_count = self.book_profiles.iloc[idx]['review_count']

                # Skip books with too few reviews
                if rec_review_count < min_rec_reviews:
                    continue

                # Check for duplicate titles
                rec_title_normalized = self.normalize_title(rec_title)
                if rec_title_normalized in seen_titles:
                    continue

                # Skip books by the same author (diversity mode always on)
                rec_author = self.books_df[self.books_df['bookid'] ==
                                           self.book_profiles.iloc[idx]['bookid']]['authors'].iloc[0]
                if pd.notna(source_author) and pd.notna(rec_author):
                    # Check if same author (compare last names)
                    source_last = str(source_author).split()[-1].lower()
                    rec_last = str(rec_author).split()[-1].lower()
                    if source_last == rec_last:
                        continue

                # Boost if same category
                if pd.notna(book_category) and pd.notna(rec_category):
                    if book_category.lower() in rec_category.lower() or \
                            rec_category.lower() in book_category.lower():
                        score += same_category_boost

                # Add to seen titles and boosted scores
                seen_titles.add(rec_title_normalized)
                boosted_scores.append((idx, score))

        # Sort by similarity (excluding the book itself)
        boosted_scores = sorted(boosted_scores, key=lambda x: x[1], reverse=True)

        # Filter by minimum similarity and get top N
        filtered_scores = [(i, s) for i, s in boosted_scores if s >= min_similarity][:n_recommendations]

        if not filtered_scores:
            print(f"No similar books found with similarity >= {min_similarity} and min {min_rec_reviews} reviews")
            return []

        # Get book indices
        book_indices = [i[0] for i in filtered_scores]
        similarity_scores = [i[1] for i in filtered_scores]

        # Create recommendations dataframe
        recommendations = self.book_profiles.iloc[book_indices].copy()
        recommendations['similarity_score'] = similarity_scores

        # Merge with full book info
        recommendations = recommendations.merge(
            self.books_df[['bookid', 'authors', 'publisher', 'avg_rating']],
            on='bookid',
            how='left'
        )

        return recommendations[['title', 'authors', 'categories', 'review_count', 'avg_rating', 'similarity_score']]

    def get_top_keywords_for_book(self, book_title, n_keywords=10):
        """Get the most important keywords for a book based on TF-IDF scores"""

        # Find the book
        book_match = self.book_profiles[
            self.book_profiles['title'].str.contains(book_title, case=False, na=False, regex=False)
        ]

        if book_match.empty:
            print(f"Book '{book_title}' not found in database")
            return []

        book_idx = book_match.index[0]

        # Get TF-IDF scores for this book
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_matrix = self.tfidf_vectorizer.transform(self.book_profiles['profile'])
        book_tfidf = tfidf_matrix[book_idx].toarray()[0]

        # Get top keywords
        top_indices = book_tfidf.argsort()[-n_keywords:][::-1]
        top_keywords = [(feature_names[i], book_tfidf[i]) for i in top_indices if book_tfidf[i] > 0]

        return top_keywords

    def search_books(self, query):
        """Search for books by title"""
        matches = self.book_profiles[
            self.book_profiles['title'].str.contains(query, case=False, na=False, regex=False)
        ]

        if matches.empty:
            print(f"No books found matching '{query}'")
            return None

        print(f"\nFound {len(matches)} book(s) matching '{query}':")
        print("-" * 80)
        for idx, row in matches.iterrows():
            print(f"{idx + 1}. {row['title']}")
            book_info = self.books_df[self.books_df['bookid'] == row['bookid']].iloc[0]
            print(f"   Author(s): {book_info['authors']}")
            print(f"   Categories: {row['categories']}")
            print(f"   Reviews: {row['review_count']}")
            print()

        return matches

    def close(self):
        """Close database connection"""
        self.engine.dispose()


# Example usage
if __name__ == "__main__":
    try:
        # Initialize the recommendation system
        recommender = BookRecommendationSystem(
            host='localhost',
            user='root',
            password='',
            database='book_reviews',
            port=3305
        )
        print("✓ Successfully connected to database!\n")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if MySQL is running")
        print("2. Verify your username and password")
        print("3. Make sure the database 'book_reviews' exists")
        print("4. Install required packages: pip install sqlalchemy mysql-connector-python")
        exit(1)

    # Try to load from cache first
    if recommender.load_cache():
        print("\n✓ Using cached data - ready for recommendations!")
    else:
        print("No cache found. Building recommendation system...")
        print("(This will take a few minutes, but only needs to be done once)\n")

        # Load data (first 10,000 books by review count, with at least 5 reviews)
        recommender.load_data(limit=10000, min_reviews=5)

        # Create book profiles from reviews (weight positive reviews more, remove character names)
        recommender.create_book_profiles(weight_positive_reviews=True, remove_names=True)

        # Calculate similarities
        recommender.calculate_similarity()

        # Save to cache for next time
        recommender.save_cache()

    # Interactive recommendation loop
    while True:
        print("\n" + "=" * 80)
        print("BOOK RECOMMENDATION SYSTEM")
        print("=" * 80)
        print("\nOptions:")
        print("1. Get recommendations for a book")
        print("2. Rebuild cache (re-process all data)")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == '3':
            break
        elif choice == '2':
            print("\nRebuilding recommendation system from database...")
            recommender.load_data(limit=10000, min_reviews=5)
            recommender.create_book_profiles(weight_positive_reviews=True, remove_names=True)
            recommender.calculate_similarity()
            recommender.save_cache()
            print("\n✓ Cache rebuilt successfully!")
        elif choice == '1':
            book_title = input("\nEnter a book title (or part of it): ").strip()

            if not book_title:
                continue

            # First, show matching books
            matches = recommender.search_books(book_title)

            if matches is None or len(matches) == 0:
                continue

            if len(matches) > 1:
                selection = input(
                    "\nEnter the number of the book to get recommendations for (or press Enter for first): ").strip()
                if selection.isdigit() and 1 <= int(selection) <= len(matches):
                    book_title = matches.iloc[int(selection) - 1]['title']
                else:
                    book_title = matches.iloc[0]['title']

            print(f"\n{'=' * 80}")
            print(f"RECOMMENDATIONS FOR: {book_title}")
            print(f"(Diversity mode enabled - excluding same author)")
            print('=' * 80)

            # Get recommendations (diversity mode always enabled)
            recommendations = recommender.get_recommendations(
                book_title,
                n_recommendations=10,
                min_similarity=0.1,
                same_category_boost=0.2,
                min_rec_reviews=20
            )

            if len(recommendations) > 0:
                print("\nRecommended Books:")
                print("-" * 80)
                for idx, row in recommendations.iterrows():
                    print(f"\n{idx + 1}. {row['title']}")
                    print(f"   Author(s): {row['authors']}")
                    print(f"   Categories: {row['categories']}")
                    print(f"   Reviews: {row['review_count']} | Avg Rating: {row['avg_rating']:.2f}")
                    print(f"   Similarity Score: {row['similarity_score']:.4f}")

                # Show top keywords for the searched book
                show_keywords = input("\nShow top keywords for this book? (y/n): ").strip().lower()
                if show_keywords == 'y':
                    print("\n" + "=" * 80)
                    print("TOP KEYWORDS FROM REVIEWS")
                    print("=" * 80)

                    keywords = recommender.get_top_keywords_for_book(book_title, n_keywords=20)
                    if keywords:
                        print(f"\nMost important keywords for '{book_title}':")
                        for keyword, score in keywords:
                            print(f"  - {keyword}: {score:.4f}")
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    # Close connection
    print("\nThank you for using the Book Recommendation System!")
    recommender.close()