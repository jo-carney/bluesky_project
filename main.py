from datasets import load_dataset
import pandas as pd
import logging
import sqlite3
from pandarallel import pandarallel
from metrics_calculator import calculate_daily_metrics, add_date_column
import re
import time
import functools

# -------------------------------------------
# Configuration
# -------------------------------------------
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Configure testing state
testing = True

# Initialize Pandarallel for parallel processing
pandarallel.initialize(nb_workers=8, progress_bar=True, use_memory_fs=False)

# Constants
DB_NAME = "posts.db"
BATCH_SIZE = 100000
if testing is True:
    MAX_BATCHES = 1
else:
    MAX_BATCHES = None

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
def timing_decorator(func):
    """
    A decorator to measure the execution time of a function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(
            f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds"
        )
        return result

    return wrapper


@timing_decorator
def preload_existing_uris(db_name=DB_NAME):
    """Preload existing URIs from the database into a set for fast lookups."""
    with sqlite3.connect(db_name) as conn:
        logging.info("Loading existing URIs from the database...")
        result = pd.read_sql_query("SELECT uri FROM posts", conn)
    return set(result["uri"].tolist())


@timing_decorator
def create_database(db_name=DB_NAME):
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                date TEXT NOT NULL,
                author TEXT NOT NULL,
                post_count INTEGER,
                replies_received INTEGER,
                matched_keywords TEXT,
                engagement_ratio REAL,
                influencer_type TEXT,
                PRIMARY KEY (date, author)
            )
            """
        )
    logging.info("Database initialized with 'posts' and 'metrics' tables.")


@timing_decorator
def drop_table(db_name=DB_NAME):
    """Drop the 'posts' and 'metrics' table if they exist."""
    with sqlite3.connect(db_name) as conn:
        conn.execute("DROP TABLE IF EXISTS posts")
        conn.execute("DROP TABLE IF EXISTS metrics")
    logging.info("Dropped 'posts' and 'metrics' table.")


@timing_decorator
def insert_posts_to_db(df, db_name=DB_NAME):
    """
    Insert posts into the database.

    Args:
        df (pd.DataFrame): DataFrame of posts to insert.
        db_name (str): Name of the SQLite database.
    """
    if df.empty:
        logging.info("No posts to insert into the database.")
        return

    # Ensure schema is correct
    try:
        validate_posts_schema(db_name)
    except ValueError as e:
        logging.exception(f"An error occurred: {e}")
        return

    try:
        with sqlite3.connect(db_name) as conn:
            df.to_sql(
                "posts", conn, if_exists="append", index=False, method="multi"
            )
            logging.info("Inserted %d posts into the database.", len(df))
    except Exception as e:
        logging.exception(f"An error occurred: {e}")


@timing_decorator
def validate_posts_schema(db_name=DB_NAME):
    """
    Ensure the 'posts' table exists with the required schema and validate it.

    Args:
        db_name (str): SQLite database file.
    """
    required_columns = {
        "uri",
        "text",
        "author",
        "reply_to",
        "created_at",
        "matched_keywords",
    }

    with sqlite3.connect(db_name) as conn:
        # Validate the schema
        cursor = conn.execute(f"PRAGMA table_info(posts)")
        existing_columns = {row[1] for row in cursor.fetchall()}

    missing_columns = required_columns - existing_columns
    if missing_columns:
        raise ValueError(
            f"Missing columns in 'posts' table: {missing_columns}"
        )
    logging.info("Schema validation passed for 'posts' table.")


@timing_decorator
def validate_metrics_schema(db_name=DB_NAME):
    """
    Ensure the 'metrics' table exists with the required schema and validate it.

    Args:
        db_name (str): SQLite database file.
    """
    required_columns = {
        "date",
        "author",
        "post_count",
        "replies_received",
        "matched_keywords",
        "engagement_ratio",
        "influencer_type",
    }

    with sqlite3.connect(db_name) as conn:
        # Validate the schema
        cursor = conn.execute(f"PRAGMA table_info(metrics)")
        existing_columns = {row[1] for row in cursor.fetchall()}

    missing_columns = required_columns - existing_columns
    if missing_columns:
        raise ValueError(
            f"Missing columns in 'metrics' table: {missing_columns}"
        )
    logging.info("Schema validation passed for 'metrics' table.")


@timing_decorator
def upsert_daily_metrics(new_metrics, db_name=DB_NAME):
    """
    Upsert (update or insert) daily metrics into the 'metrics' table.

    Args:
        new_metrics (pd.DataFrame): DataFrame containing daily metrics to upsert.
        db_name (str): SQLite database name.
    """
    logging.info("Upserting daily metrics into the database...")

    # Ensure schema is correct
    try:
        validate_metrics_schema(db_name)
    except ValueError as e:
        logging.exception(f"An error occurred: {e}")
        return

    with sqlite3.connect(db_name) as conn:
        for _, row in new_metrics.iterrows():
            try:
                conn.execute(
                    """
                    INSERT INTO metrics (date, author, post_count, replies_received, matched_keywords, engagement_ratio, influencer_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(date, author) DO UPDATE SET
                        post_count=excluded.post_count,
                        replies_received=excluded.replies_received,
                        matched_keywords=excluded.matched_keywords,
                        engagement_ratio=excluded.engagement_ratio,
                        influencer_type=excluded.influencer_type
                    """,
                    (
                        row["date"],
                        row["author"],
                        row["post_count"],
                        row["replies_received"],
                        row["matched_keywords"],
                        row["engagement_ratio"],
                        row["influencer_type"],
                    ),
                )
            except Exception as e:
                logging.exception(f"An error occurred: {e}")

    logging.info("Daily metrics upserted successfully.")


@timing_decorator
def find_keywords(text_lower):
    found = [kw for kw in APPLE_KEYWORDS if kw in text_lower]
    return ", ".join(found) if found else ""


@timing_decorator
def process_batch(batch, existing_uris):
    """Process a batch of data: filter duplicates and match Apple-related keywords."""
    df = pd.DataFrame(batch)
    logging.info("Filtering out existing URIs...")
    df = df[~df["uri"].isin(existing_uris)].copy()
    if df.empty:
        logging.info("No new URIs to process in this batch.")
        return df

    # Replace the old regex-based keyword matching with the substring approach
    logging.info("Matching Apple-related keywords...")
    df["text_lower"] = df["text"].str.lower()  # Convert once
    df["matched_keywords"] = df["text_lower"].apply(find_keywords)
    df.drop(columns=["text_lower"], inplace=True)

    df = df[df["matched_keywords"] != ""].copy()
    logging.info(
        "Filtered to %d posts containing Apple-related keywords.", len(df)
    )
    return df
    return df


# -------------------------------------------
# Main ETL Pipeline
# -------------------------------------------
if __name__ == "__main__":
    overall_start = time.time()
    # Drop and recreate the database
    drop_table()
    create_database()

    # Load dataset
    logging.info("Loading dataset from Hugging Face...")
    dataset = load_dataset(
        "alpindale/two-million-bluesky-posts", split="train", streaming=True
    )

    # Preload existing URIs
    existing_uris = preload_existing_uris(DB_NAME)

    total_batches = 0
    apple_posts_count = 0
    batch = []

    # Process dataset in batches
    for row in dataset:
        batch.append(
            {
                "uri": row["uri"],
                "text": row["text"],
                "author": row["author"],
                "reply_to": row["reply_to"],
                "created_at": row["created_at"],
            }
        )
        if len(batch) >= BATCH_SIZE:
            logging.info(f"Processing batch {total_batches + 1}...")
            df_filtered = process_batch(batch, existing_uris)
            if not df_filtered.empty:
                insert_posts_to_db(df_filtered)
                apple_posts_count += len(df_filtered)
                existing_uris.update(df_filtered["uri"])
            batch.clear()
            total_batches += 1
            if MAX_BATCHES is not None and total_batches >= MAX_BATCHES:
                logging.info("Max batch limit reached. Stopping processing.")
                break

    logging.info(
        f"ETL pipeline completed. Total Apple-related posts: {apple_posts_count}."
    )

    # Fetch posts and calculate metrics
    with sqlite3.connect(DB_NAME) as conn:
        df_posts = pd.read_sql_query("SELECT * FROM posts", conn)
    # Calculate daily metrics
    if not df_posts.empty:
        logging.info("Calculating and upserting daily metrics...")

        df_posts = add_date_column(df_posts)
        daily_metrics = calculate_daily_metrics(df_posts)

        # Upsert metrics for each day into the database
        upsert_daily_metrics(daily_metrics)

        logging.info(
            f"ETL pipeline completed in {time.time() - overall_start:.4f} seconds"
        )
    else:
        logging.warning("No posts found in the database to calculate metrics.")
