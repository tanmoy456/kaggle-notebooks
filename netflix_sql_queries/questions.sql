-- ==========================================
-- Netflix Data Exploration Queries
-- Database: netflix_db
-- Table: netflix_titles
-- ==========================================

USE netflix_db;

-- ==========================================
-- Q1: Count Movies vs TV Shows
-- ==========================================
SELECT type, COUNT(*) AS count
FROM netflix_titles
GROUP BY type;
/*
  Notes:
  COUNT(*) counts all rows regardless of NULLs.
  COUNT(type) gives the same result here because
  the type column has no NULLs — every row is
  either 'Movie' or 'TV Show'.
*/

-- ==========================================
-- Q2: Most common rating for each type
-- ==========================================
SELECT type, rating, COUNT(*) AS count
FROM netflix_titles
GROUP BY type, rating
ORDER BY type, count DESC;

-- ==========================================
-- Q3: List all movies released in a specific year (e.g., 2020)
-- ==========================================

SELECT *
FROM netflix_titles
-- WHERE release_year IN (2020)
 WHERE release_year = 2020;

-- ==========================================
-- Q4: Find the top 5 countries with the most content on Netflix
-- ==========================================

SELECT country, COUNT(*) AS count
FROM netflix_titles
WHERE country IS NOT NULL AND country != ''
GROUP BY country
ORDER BY count DESC
LIMIT 5;

-- ==========================================
-- Q5: Identify the longest movie
-- ========================================== 
/*Problem: duration column stores "90 min" as text

We need the number part to compare and sort
Step 1: REPLACE strips the text
         "90 min" → "90"   (still a string)
Step 2: CAST converts to a real number
         "90" → 90         (now a number)
*/

-- SELECT title, duration, REPLACE(duration, ' min', '') AS after_replace
SELECT title, type, duration, CAST(REPLACE(duration, ' min', '') AS UNSIGNED) AS minutes
-- SELECT *
FROM netflix_titles
WHERE type IN ('Movie')
ORDER BY  minutes DESC
LIMIT 5;

-- ==========================================
-- Q6: Which 10 years had the most content released on Netflix?
-- ==========================================

SELECT release_year, COUNT(*) AS total
FROM netflix_titles
WHERE type IN ('Movie', 'TV Show')
GROUP BY release_year
ORDER BY total DESC
LIMIT 10;

-- ==========================================
-- Q7: Find content added in the last 5 years
-- ==========================================

/*
  Q6: Find content added in the last 5 years

  Problem: date_added is a string like "September 25, 2021"
  MySQL can't compare strings as dates, so we convert first.

  STR_TO_DATE(date_added, '%M %d, %Y')
    Converts string → real date
    'September 25, 2021' → 2021-09-25
      %M = full month name (September)
      %d = day number (25)
      %Y = 4-digit year (2021)

  CURDATE()
    Returns today's date → 2026-06-14

  DATE_SUB(CURDATE(), INTERVAL 5 YEAR)
    Subtracts 5 years from today → 2021-06-14

  >= combines them:
    Keep rows where converted date >= 2021-06-14
    2021-09-25 >= 2021-06-14  >> kept
    2019-03-10 >= 2021-06-14  >> filtered out

  If date_added was stored as a DATE type instead
  of a string, we could skip STR_TO_DATE entirely:
    WHERE date_added >= DATE_SUB(CURDATE(), INTERVAL 5 YEAR)
*/

SELECT title, type, date_added
FROM netflix_titles
WHERE STR_TO_DATE(date_added, '%M %d, %Y') >= DATE_SUB(CURDATE(), INTERVAL 5 YEAR);


-- ==========================================
-- Q8: Find all movies/TV shows by director 'Rajiv Chilaka'
-- ==========================================

SELECT title, type, director
FROM netflix_titles
-- WHERE director IN ('Rajiv Chilaka')
WHERE director LIKE '%Rajiv Chilaka%';

-- ==========================================
-- Q10: List all movies that are documentaries
-- ==========================================

SELECT *
FROM netflix_titles
WHERE type IN ('Movie') AND listed_in LIKE '%docu%';

