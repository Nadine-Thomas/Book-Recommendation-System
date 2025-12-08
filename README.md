# Book Review Analysis & Recommendation System 
### Team: Nadine Thomas, Gracie Lovell, and Kacie Myers

## Overview
This project builds a book recommendation system by combining SQL data with review text. Using text mining techniques, the system identifies themes, sentiment patterns, and textual similarities between books. The final recommendation engine ranks books based on similarity scores, quality filters, and category relavance.

## Features
**1. Relational Database (MySQL)**
  - Stores books, reviews, and associated metadata
  - Includes indexes for fast lookups
  - Supports importing large CSV datasets
**2. Text Mining (Python)**
  - Processes and cleans review text
  - Builds a profile for each book title
  - Uses Term Frequency - Inverse Document Frequency (TF-IDF) to vectorize the review text
  - Computes cosine similarity between unique books
**3. Recommendation Engine**
  - Generates a similarity score for every book
  - Applies quality filters (minimum review count)
  - Excludes books from the same author
  - Slightly boosts similarity score for books of the same genre
  - Returns top recommendations based on all factors

## Project Structure
```
├── data/
│   ├── books_data.csv
│   ├── amazon_book_reviews.csv
├── sql/
│   └── databasesetup.sql
├── src/
│   ├── main.py
├── models/
│   ├── book_recommender_cache.pkl
├── README.md
├── user_manual.pdf
└── developers_manual.pdf
```

## Technology Stack
**Python Libraries**
- ```pandas``` - data manipulation
- ```scikit-learn``` - TF-IDF vectorization and cosine similarity
- ```re``` - text cleaning
- ```collections.defaultdict``` - structure accumulation of data
- ```sqlalchemy``` - connection to database and queries
- ```pickle``` - model persistance
- ```os``` - directory/file handling

**Database**
- MySQL Community Server
- Imported CSV datasets from kaggle.com

## How it Works
### 1. Load & Clean Data
- Books and reviews are imported in MySQL database. Python pulls data into DataFrames for processing.
### 2. Build Book Profiles
- Each book is summarized into a single document built from reader reviews, book description, and categories/genres.
### 3. TF-IDF Vectorization
- TF-IDF converts each profile into a numerical vector that highlights unique and descriptive keywords.
### 4. Cosine Similarity
- Every vector is compared to the source book, producing a similarity score between 0 and 1.
### 5. Post-Processing Filters & Boosts
- Books with <20 reviews or books written by the same author are filtered out. Genre/category matches recieve a small boost.
### 6. Final Recommendation Report
- The system outputs the most similar books based on TF-IDF and refined scoring methods.

## How to Run
### Requirements
- Software:
  - Python 3.9 or later
  - SQL database software (MySQLWorkbench preferred)
  - Python editor (VSCode preferred)
- Python Libraries:
  - Install the required libraries by running the following command in a terminal:
  ```
  pip install pandas numpy nltk scikit-learn sqlalchemy
  ```
  - Additional download for sentiment analysis:
  ```
  import nltk
  nltk.download('vader_lexicon')
  ```
- Data Files:
  - ```amazon_book_reviews.csv``` - contains book ratings and review text
  - ```books_data.csv``` - contains book metadata
  - ```databasesetup.sql``` - SQL code used to combine two CSV files
  - ```main.py``` - Python code that performs text analysis and provides recommendations
  
### Step 1: Run SQL File
1. Open SQL Program
2. Open the ```databasesetup.sql``` file
3. Run the script

This script creates the two tables in the database and loads all book information into their respective tables. After the script runs, you should have a single database called book_reviews with two tables: one containing rating, review text, and book title, and the other containing all book information. These two tables will share an indentifer key called bookid.

### Step 2: Run Python File
1. Open a Terminal or Python Editor
2. Run the following command:
```
python main.py
```
3. The first run of this file is expected to take about 20 minutes. This will cache all of the book profiles into a PKL file called book_recommender_cache.pkl. From now on, succeeding runs should only take a couple of seconds.

This file connects to the SQL database and prompts the user to enter a book for the recommendation system.

## Documentation
- **Developer's Manual:** detailed technical explanation of the system architecture
  - Link: [Developer's Manual](https://github.com/Nadine-Thomas/Book-Recommendation-System/blob/main/developers_manual.pdf)
- **User Manual:** instructions for navigating the interface and troubleshooting issues
  - Link: [User Manual](https://github.com/Nadine-Thomas/Book-Recommendation-System/blob/main/user_manual.pdf)
