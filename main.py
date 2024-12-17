from datasets import load_dataset
import pandas as pd
import logging
from metrics_calculator import calculate_daily_metrics, add_date_column
import sqlite3
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import re
import spacy


# Configure info-level logging and format for cleaner output
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Ensure consistent language detection results
DetectorFactory.seed = 42

# Load the English NLP pipeline
nlp = spacy.load("en_core_web_sm")

# Define Apple-related keywords
KEYWORDS = {
    "products": [
        "iphone",
        "iphones",
        "ipad",
        "ipad pro",
        "macbook",
        "mac",
        "mac mini",
        "mac studio",
        "airpods",
        "apple watch",
        "watch ultra",
        "apple tv",
        "vision pro",
        "apple pencil",
    ],
    "software": [
        "ios",
        "ipados",
        "macos",
        "safari",
        "facetime",
        "siri",
        "icloud",
        "airdrop",
    ],
    "chips": ["m1", "m2", "m3", "a17", "a16", "apple silicon"],
    "branding": [
        "apple",
        "tim cook",
        "apple store",
        "apple music",
        "apple care",
    ],
}


# -------------------------------------------
# Text Preprocessing
# -------------------------------------------
def preprocess_texts_in_batch(texts):
    """
    Preprocess a list of texts in a batch using Spacy pipeline.
    Args:
        texts (list): List of raw text strings.
    Returns:
        list: Preprocessed text strings.
    """
    docs = list(
        nlp.pipe(texts, disable=["ner", "parser"])
    )  # Disable unneeded components
    return [
        " ".join(
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and len(token.text) > 2
        )
        for doc in docs
    ]


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


def filter_apple_posts(df, keywords):
    """
    Filter posts containing specific Apple-related keywords.
    Use a regex pattern for efficient matching.
    """
    keyword_pattern = "|".join(re.escape(keyword) for keyword in keywords)
    apple_posts = df[
        df["text"].str.contains(keyword_pattern, case=False, na=False)
    ]
    logging.info("%d posts contain Apple-related keywords.", len(apple_posts))
    return apple_posts


# -------------------------------------------
# Apple Keyword Filtering
# -------------------------------------------
def categorize_posts_by_keywords(text, keyword_groups):
    """
    Categorize text based on predefined keyword groups.
    """
    matched_categories = []
    text_lower = text.lower()
    for category, keywords in keyword_groups.items():
        if any(keyword in text_lower for keyword in keywords):
            matched_categories.append(category)
    return matched_categories


def filter_and_categorize_apple_posts(df, keyword_groups):
    """
    Filter posts containing Apple-related keywords and categorize them.

    Args:
        df (pd.DataFrame): DataFrame containing 'text' column.
        keyword_groups (dict): Dictionary of keyword categories.

    Returns:
        pd.DataFrame: Filtered and categorized posts.
    """
    keyword_regex = "|".join(
        re.escape(keyword) for keyword in sum(keyword_groups.values(), [])
    )

    # Filter posts containing Apple-related keywords
    df_filtered = df[
        df["text"].str.contains(keyword_regex, case=False, na=False)
    ]

    # Categorize posts
    def join_categories(text):
        categories = categorize_posts_by_keywords(text, keyword_groups)
        return ", ".join(categories)  # Convert list to comma-separated string

    df_filtered["categories"] = df_filtered["text"].apply(join_categories)
    return df_filtered


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
            language TEXT,
            categories TEXT
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
    max_batches = 5  # Remove limit on max_batches
    batch = []
    total_batches = 0
    total_posts = 0
    apple_posts_count = 0

    for i, row in enumerate(dataset):
        # Preprocess text before language detection
        texts = [row["text"] for row in dataset]
        preprocessed_texts = preprocess_texts_in_batch(texts)

        batch.append(
            {
                "uri": row["uri"],
                "text": preprocess_texts_in_batch,
                "author": row["author"],
                "reply_to": row["reply_to"],
                "created_at": row["created_at"],
                "language": detect_language(preprocess_texts_in_batch),
            }
        )

        # Process and insert the batch when full
        if len(batch) == batch_size:
            logging.info("Processing batch %d...", total_batches + 1)
            df = pd.DataFrame(batch)

            # Filter for English posts
            df = filter_english_posts(df)

            # Filter for Apple-related posts
            df_apple = filter_and_categorize_apple_posts(df, KEYWORDS)

            # Write Apple-related posts to the database
            if not df_apple.empty:
                insert_posts_to_db(df_apple)
                apple_posts_count += len(df_apple)

            batch = []  # Clear batch
            total_batches += 1
            logging.info(
                f"Processed {total_batches} batches, {apple_posts_count} Apple-related posts saved."
            )

    # Process remaining rows
    if batch:
        df = pd.DataFrame(batch)
        df = filter_english_posts(df)
        df_apple = filter_and_categorize_apple_posts(df, KEYWORDS)
        if not df_apple.empty:
            insert_posts_to_db(df_apple)
            apple_posts_count += len(df_apple)

    logging.info(
        f"ETL pipeline completed. Total Apple-related posts: {apple_posts_count}."
    )

    # Reload data and calculate daily metrics
    conn = sqlite3.connect("posts.db")
    df_posts = pd.read_sql_query("SELECT * FROM posts", conn)
    conn.close()

    # Add date column and calculate metrics
    df_posts = add_date_column(df_posts)
    if not df_posts.empty:
        logging.info("Calculating daily metrics for Apple-related posts...")
        daily_metrics = calculate_daily_metrics(df_posts)
        print(daily_metrics.sample(10))
