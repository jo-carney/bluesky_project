from datasets import load_dataset
import pandas as pd
import logging
import sqlite3
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from metrics_calculator import calculate_daily_metrics, add_date_column
import re

# -------------------------------------------
# Configuration
# -------------------------------------------
# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Language detection consistency
DetectorFactory.seed = 42

# Constants
DB_NAME = "posts.db"
TABLE_NAME = "posts"
BATCH_SIZE = 500
MAX_BATCHES = 10

# Apple-related keywords
APPLE_KEYWORDS = [
    "iphone",
    "ipad",
    "ipad pro",
    "macbook",
    "mac mini",
    "mac studio",
    "airpods",
    "apple watch",
    "watch ultra",
    "apple tv",
    "vision pro",
    "apple pencil",
    "ios",
    "ipados",
    "macos",
    "safari",
    "facetime",
    "siri",
    "icloud",
    "airdrop",
    "m1",
    "m2",
    "m3",
    "a17",
    "a16",
    "apple silicon",
    "tim cook",
    "apple store",
    "apple music",
    "apple care",
]

# Precompile keyword regex
APPLE_KEYWORDS_REGEX = re.compile(
    r"|".join(rf"\b{re.escape(kw)}\b" for kw in APPLE_KEYWORDS), re.IGNORECASE
)

# -------------------------------------------
# Functions
# -------------------------------------------


def detect_language(text):
    """
    Detect the language of a given text using langdetect.

    Args:
        text (str): Input text string.

    Returns:
        str: 'en' if detected as English; otherwise None.
    """
    try:
        return "en" if detect(text) == "en" else None
    except LangDetectException:
        return None


def filter_english_and_apple_posts(df, keyword_regex):
    """
    Filter posts to include only English-language posts with Apple-related keywords.

    Args:
        df (pd.DataFrame): Input DataFrame with 'text' column.
        keyword_regex (re.Pattern): Compiled regex for matching Apple-related keywords.

    Returns:
        pd.DataFrame: Filtered DataFrame containing English Apple-related posts.
    """
    logging.info("Filtering posts for English language...")
    df = df[
        df["text"].apply(lambda text: detect_language(text) == "en")
    ].copy()

    def find_keywords(text):
        matches = re.findall(keyword_regex, text)
        return ", ".join(
            set(matches)
        )  # Return unique matched keywords as a string

    logging.info("Filtering posts for Apple-related keywords...")
    df["matched_keywords"] = df["text"].apply(
        find_keywords
    )  # Add matched keywords column
    df = df[
        df["matched_keywords"] != ""
    ].copy()  # Ensure no SettingWithCopyWarning

    return df


def remove_duplicate_uris(df):
    """
    Remove duplicate 'uri' values within a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing posts.

    Returns:
        pd.DataFrame: DataFrame with duplicate 'uri' values removed.
    """
    df = df.drop_duplicates(subset=["uri"])
    return df


def filter_existing_uris(df, db_name="posts.db"):
    """
    Remove rows with 'uri' values already present in the database.

    Args:
        df (pd.DataFrame): DataFrame to filter.
        db_name (str): SQLite database file name.

    Returns:
        pd.DataFrame: Filtered DataFrame excluding existing 'uri' values.
    """
    with sqlite3.connect(db_name) as conn:
        existing_uris = pd.read_sql_query("SELECT uri FROM posts", conn)[
            "uri"
        ].tolist()
    return df[~df["uri"].isin(existing_uris)]


def drop_table(db_name="posts.db"):
    """
    Drop the 'posts' table if it exists in the database.

    Args:
        db_name (str): Name of the SQLite database file.
    """
    with sqlite3.connect(db_name) as conn:
        conn.execute("DROP TABLE IF EXISTS posts")
    logging.info("Dropped 'posts' table.")


def create_database(db_name="posts.db"):
    """
    Create a SQLite database and initialize the 'posts' table.

    Args:
        db_name (str): Name of the SQLite database file.
    """
    with sqlite3.connect(db_name) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS posts (
                uri TEXT PRIMARY KEY,
                text TEXT,
                author TEXT,
                reply_to TEXT,
                created_at TEXT,
                matched_keywords TEXT
            )
            """
        )
    logging.info("Database initialized with 'posts' table.")


def insert_posts_to_db(df, db_name="posts.db"):
    """
    Insert posts into the database while ensuring unique URIs.

    Args:
        df (pd.DataFrame): DataFrame of posts to insert.
        db_name (str): Name of the SQLite database.
    """
    if df.empty:
        logging.info("No posts to insert into the database.")
        return

    with sqlite3.connect(db_name) as conn:
        df.to_sql(
            "posts", conn, if_exists="append", index=False, method="multi"
        )
    logging.info("Inserted %d posts into the database.", len(df))


# -------------------------------------------
# Main ETL Pipeline
# -------------------------------------------

if __name__ == "__main__":
    # Initialize the database
    drop_table()
    create_database()

    # Load dataset
    logging.info("Loading dataset from Hugging Face...")
    dataset = load_dataset(
        "alpindale/two-million-bluesky-posts", split="train", streaming=True
    )

    # Batch processing configuration
    batch_size = 500
    total_batches = 0
    apple_posts_count = 0
    batch = []

    # Main loop to process dataset
    for row in dataset:
        # Append row to batch
        batch.append(
            {
                "uri": row["uri"],
                "text": row["text"],
                "author": row["author"],
                "reply_to": row["reply_to"],
                "created_at": row["created_at"],
            }
        )

        # Process when batch size is reached
        if len(batch) >= BATCH_SIZE:
            logging.info(f"Processing batch {total_batches + 1}...")
            df = pd.DataFrame(batch)

            # Step 1: Filter English and Apple-related posts
            df_filtered = filter_english_and_apple_posts(
                df, APPLE_KEYWORDS_REGEX
            )

            # Step 2: Remove duplicates and filter existing URIs
            df_filtered = remove_duplicate_uris(df_filtered)
            df_filtered = filter_existing_uris(df_filtered)

            # Step 3: Insert into database
            insert_posts_to_db(df_filtered)

            apple_posts_count += len(df_filtered)
            batch.clear()
            total_batches += 1
            logging.info(
                f"Total Apple-related posts so far: {apple_posts_count}"
            )

            if MAX_BATCHES is not None and total_batches >= MAX_BATCHES:
                logging.info("Max batch limit reached. Stopping processing.")
                break

    # Log final summary
    logging.info(
        f"ETL pipeline completed. Total Apple-related posts: {apple_posts_count}."
    )

    # Fetch posts from the database
    with sqlite3.connect("posts.db") as conn:
        df_posts = pd.read_sql_query("SELECT * FROM posts", conn)

    # Calculate and display metrics
    if not df_posts.empty:
        logging.info("Calculating daily metrics for Apple-related posts...")
        # Add date column if not already added
        daily_metrics = calculate_daily_metrics(df_posts)

        logging.info("\n--- Daily Influencer Metrics ---\n")
        print(daily_metrics.head(10))  # Display top 10 rows of metrics
    else:
        logging.warning("No posts found in the database to calculate metrics.")
