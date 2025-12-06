# create and use the database
CREATE DATABASE book_reviews;
USE book_reviews;

# enables the local infile on the server
SET GLOBAL local_infile = 1;
SHOW VARIABLES LIKE 'local_infile';

# creates the books table
DROP TABLE IF EXISTS books;
CREATE TABLE books (
	bookid INT AUTO_INCREMENT PRIMARY KEY,
	title VARCHAR(500),
    description TEXT,
    authors VARCHAR(500),
    publisher VARCHAR(500),
    publishdate DATE,
    categories VARCHAR(500)
);

# load the books_data csv file into the table
LOAD DATA LOCAL INFILE ''
INTO TABLE books
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(title, description, authors, publisher, publishdate, categories);

# creates the reviews table
DROP TABLE IF EXISTS reviews;
CREATE TABLE reviews (
	reviewid INT AUTO_INCREMENT PRIMARY KEY,
    bookid INT,
	title VARCHAR(500),
    reviewscore INT,
    reviewsummary VARCHAR(100),
    reviewtext TEXT,
    FOREIGN KEY (bookid) REFERENCES books(bookid)
);

# loads the amazon_book_reviews csv file into the table
LOAD DATA LOCAL INFILE ''
INTO TABLE reviews
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(title, reviewscore, reviewsummary, reviewtext);

SET SESSION wait_timeout = 3000;
SET SESSION interactive_timeout = 3000;
SET SESSION net_read_timeout = 1800;

ALTER TABLE books ADD INDEX idx_books_title (title);
ALTER TABLE reviews ADD INDEX idx_reviews_title (title);

# add the corresponding bookid to each reviewid
UPDATE reviews r
JOIN books b ON r.title = b.title
SET r.bookid = b.bookid;

DELETE FROM reviews
WHERE bookid IS NULL;

select * from reviews;
select * from books;
