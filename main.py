from datasets import load_dataset
import pandas as pd
import logging
from metrics_calculator import calculate_daily_metrics, add_date_column
import sqlite3
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import re

# Configure info-level logging and format for cleaner output
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Ensure consistent language detection results
DetectorFactory.seed = 42


# -------------------------------------------
# Text Preprocessing
# -------------------------------------------
def normalize_whitespace(text):
    """
    Normalize whitespace in the text by replacing multiple spaces or newlines with a single space.

    Args:
        text (str): Input text.

    Returns:
        str: Text with normalized whitespace.
    """
    return " ".join(text.split())


def remove_special_characters(text):
    """
    Remove special characters from the text.

    Args:
        text (str): Input text.

    Returns:
        str: Text without special characters.
    """
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)


def to_lowercase(text):
    """
    Convert text to lowercase.

    Args:
        text (str): Input text.

    Returns:
        str: Text in lowercase.
    """
    return text.lower()


def preprocess_text(text):
    """
    Apply preprocessing steps to clean the input text.

    Args:
        text (str): Input text.

    Returns:
        str: Preprocessed text.
    """
    # text = remove_emojis(text)
    text = remove_special_characters(text)
    text = normalize_whitespace(text)
    # text = remove_urls(text)
    # text = remove_mentions_and_hashtags(text)
    text = to_lowercase(text)
    # text = remove_stopwords(text)
    # text = remove_non_ascii(text)
    return text


# -------------------------------------------
# Language Detection
# -------------------------------------------
def detect_language(text):
    """
    Detect the language of a given text using langdetect.

    Args:
        text (str): Input text.

    Returns:
        str: Detected language code (e.g., 'en', 'fr') or None if detection fails.
    """
    try:
        return detect(text)
    except LangDetectException:
        logging.warning("Failed to detect language for text: %s", text)
        return None


def filter_english_posts(df):
    """
    Filter DataFrame to include only English posts.

    Args:
        df (pd.DataFrame): Input DataFrame with 'text' column.

    Returns:
        pd.DataFrame: Filtered DataFrame with English posts.
    """
    logging.info("Detecting language for %d posts...", len(df))
    df["language"] = df["text"].apply(detect_language)
    english_posts = df[df["language"] == "en"]
    logging.info("%d English posts identified.", len(english_posts))
    return english_posts


# -------------------------------------------
# Database Operations
# -------------------------------------------
def create_posts_table(db_name="posts.db"):
    """
    Create the posts table if it doesn't exist.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS posts (
            uri TEXT PRIMARY KEY,
            text TEXT,
            author TEXT,
            reply_to TEXT,
            created_at TEXT,
            language TEXT
        )
    """
    )
    conn.commit()
    conn.close()


def insert_posts_to_db(df, db_name="posts.db"):
    """
    Insert a batch of posts into the SQLite database.
    """
    conn = sqlite3.connect(db_name)
    df.to_sql("posts", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    logging.info("Inserted %d posts into the database.", len(df))


def drop_tables(db_name="posts.db"):
    """
    Drop the posts table if it exists.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS posts")
    conn.commit()
    conn.close()
    logging.info("Posts table dropped.")


# -------------------------------------------
# Main ETL Pipeline
# -------------------------------------------
if __name__ == "__main__":
    drop_tables()
    create_posts_table()

    logging.info("Starting the ETL pipeline...")

    # Load the dataset
    logging.info("Loading dataset from Hugging Face...")
    dataset = load_dataset(
        "alpindale/two-million-bluesky-posts", split="train", streaming=True
    )

    # Process the dataset in batches
    batch_size = 1000
    max_batches = 2  # Limit to 2 batches for testing
    batch = []
    total_batches = 0
    total_posts = 0

    for i, row in enumerate(dataset):
        # Preprocess text before language detection
        preprocessed_text = preprocess_text(row["text"])

        batch.append(
            {
                "uri": row["uri"],
                "text": preprocessed_text,  # Save preprocessed text
                "author": row["author"],
                "reply_to": row["reply_to"],
                "created_at": row["created_at"],
                "language": detect_language(
                    preprocessed_text
                ),  # Detect language after preprocessing
            }
        )

        # Process and insert the batch when full
        if len(batch) == batch_size:
            logging.info("Processing batch %d...", total_batches + 1)
            df = pd.DataFrame(batch)

            # Debugging: Log created_at column samples
            logging.debug(
                "Sample 'created_at' values before processing: %s",
                df["created_at"].head(),
            )
            logging.debug(
                "Data types in 'created_at': %s", df["created_at"].dtype
            )

            # Filter for English posts
            df = filter_english_posts(df)

            # Write English posts to the database
            if not df.empty:
                insert_posts_to_db(df)
                total_posts += len(df)

            batch = []  # Clear the batch for the next set of rows
            total_batches += 1

            # Stop processing if max_batches is reached
            if total_batches >= max_batches:
                logging.info(
                    "Reached batch limit of %d. Stopping pipeline.",
                    max_batches,
                )
                break

    # Process any remaining rows
    if batch and total_batches < max_batches:
        logging.info("Processing final batch...")
        df = pd.DataFrame(batch)
        df = filter_english_posts(df)
        if not df.empty:
            insert_posts_to_db(df)
            total_posts += len(df)

    logging.info(
        "ETL pipeline completed. Processed %d batches and inserted %d posts into the database.",
        total_batches,
        total_posts,
    )

    # Step 2: Reload data from the database
    logging.info("Loading posts from the database for metrics calculation...")
    conn = sqlite3.connect("posts.db")
    df_posts = pd.read_sql_query("SELECT * FROM posts", conn)
    conn.close()

    # Step 3: Calculate daily metrics
    df_posts = add_date_column(df_posts)

    if not df_posts.empty:
        logging.info("Calculating daily influencer metrics...")
        daily_metrics = calculate_daily_metrics(df_posts)

        # Display a sample of daily metrics
        logging.info("Sample of Daily Metrics:")
        print(daily_metrics.sample(10))
    else:
        logging.warning(
            "No posts were found in the database to calculate metrics."
        )
