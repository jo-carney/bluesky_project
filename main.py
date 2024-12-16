from datasets import load_dataset
import pandas as pd
from datetime import datetime
import numpy as np
import logging

# Configure info-level logging and format for cleaner output
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# -------------------------------------------
# Data Loading and Preparation
# -------------------------------------------
def load_influencer_data(sample_size=10000):
    """
    Load influencer data from the Bluesky dataset.

    Args:
        sample_size (int): Number of rows to load.

    Returns:
        pd.DataFrame: DataFrame of posts with relevant fields.
    """
    logging.info("Loading data...")
    dataset = load_dataset(
        "alpindale/two-million-bluesky-posts", split="train", streaming=True
    )
    data = [
        {
            "uri": row["uri"],
            "text": row["text"],
            "author": row["author"],
            "reply_to": row["reply_to"],
            "created_at": datetime.fromisoformat(row["created_at"]),
        }
        for i, row in enumerate(dataset)
        if i < sample_size
    ]
    logging.info(f"Loaded {len(data)} posts.")
    return pd.DataFrame(data)


def add_date_column(df):
    """
    Add a 'date' column extracted from the 'created_at' timestamp.

    Args:
        df (pd.DataFrame): DataFrame containing the 'created_at' column.

    Returns:
        pd.DataFrame: Updated DataFrame with a 'date' column.
    """
    if not pd.api.types.is_datetime64_any_dtype(df["created_at"]):
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    df = df.dropna(subset=["created_at"])

    # Add the date column
    df.loc[:, "date"] = df["created_at"].dt.date
    return df


def create_post_author_lookup(df):
    """
    Create a lookup dictionary mapping post URIs to authors.

    Args:
        df (pd.DataFrame): DataFrame of posts.

    Returns:
        dict: Mapping of post URIs to authors.
    """
    return df.set_index("uri")["author"].to_dict()


def map_replies_to_authors(df, post_author_lookup):
    """
    Map replies to their original post authors.

    Args:
        df (pd.DataFrame): DataFrame of posts.
        post_author_lookup (dict): Mapping of post URIs to authors.

    Returns:
        pd.DataFrame: Updated DataFrame with 'original_author' column.
    """
    df.loc[:, "original_author"] = df["reply_to"].map(post_author_lookup)
    return df


# -------------------------------------------
# Metric Calculations
# -------------------------------------------
def calculate_post_count(df):
    """Calculate the number of posts made by each author."""
    return df["author"].value_counts().reset_index(name="post_count")


def calculate_replies_received(df):
    """Calculate the number of replies received by each author."""
    replies_received = df["original_author"].value_counts().reset_index()
    replies_received.columns = ["author", "replies_received"]
    return replies_received


def calculate_audience_builders(df):
    """
    Calculate unique replies per original post.

    Args:
        df (pd.DataFrame): DataFrame containing posts.

    Returns:
        pd.DataFrame: DataFrame with unique reply counts per author.
    """
    replies = df[df["reply_to"].notna()]
    unique_replies = (
        replies.groupby("original_author")["author"].nunique().reset_index()
    )
    unique_replies.columns = ["author", "unique_replies"]
    return unique_replies


def categorize_influencers(metrics):
    """
    Categorize influencers based on post activity and replies received.

    Args:
        metrics (pd.DataFrame): DataFrame with influencer metrics.

    Returns:
        pd.DataFrame: DataFrame with influencer categories.
    """
    conditions = [
        (metrics["post_count"] > metrics["post_count"].median())
        & (metrics["replies_received"] > metrics["replies_received"].median()),
        (metrics["post_count"] <= metrics["post_count"].median())
        & (metrics["replies_received"] > metrics["replies_received"].median()),
        (metrics["post_count"] > metrics["post_count"].median())
        & (
            metrics["replies_received"] <= metrics["replies_received"].median()
        ),
    ]
    categories = ["Influential Creator", "Thought Leader", "Broadcaster"]
    metrics["category"] = np.select(
        conditions, categories, default="General User"
    )
    return metrics


def calculate_daily_metrics(df):
    """
    Calculate daily influencer metrics, including post count, replies received,
    engagement ratio, audience builders, and influencer categories.

    Args:
        df (pd.DataFrame): DataFrame containing posts data.

    Returns:
        pd.DataFrame: A DataFrame with daily metrics for each user.
    """
    # Add date column
    df = add_date_column(df)

    # Create a post-to-author lookup
    post_author_lookup = create_post_author_lookup(df)

    # Map replies to their original authors
    df = map_replies_to_authors(df, post_author_lookup)

    # Initialize an empty list to store daily metrics
    daily_metrics = []

    # Group data by 'date'
    for date, group in df.groupby("date"):
        # Calculate post count
        post_count = calculate_post_count(group)

        # Calculate replies received
        replies_received = calculate_replies_received(group)

        # Merge metrics
        metrics = pd.merge(
            post_count, replies_received, on="author", how="outer"
        ).fillna(0)
        metrics["engagement_ratio"] = (
            metrics["replies_received"] / metrics["post_count"]
        )

        # Calculate audience builders
        audience_builders = calculate_audience_builders(group)

        # Merge audience builders into metrics
        metrics = pd.merge(
            metrics, audience_builders, on="author", how="left"
        ).fillna(0)

        # Categorize influencers
        metrics = categorize_influencers(metrics)

        # Add the date column to the results
        metrics["date"] = date

        # Append to results
        daily_metrics.append(metrics)

    # Combine all daily results into a single DataFrame
    return pd.concat(daily_metrics, ignore_index=True)


# -------------------------------------------
# Main Execution
# -------------------------------------------
if __name__ == "__main__":
    # Step 1: Load data
    df_posts = load_influencer_data(sample_size=10000)

    # Step 2: Calculate daily metrics
    logging.info("Calculating daily influencer metrics...")
    daily_metrics = calculate_daily_metrics(df_posts)

    # Display a sample of the results
    logging.info("Sample of Daily Metrics:")
    print(daily_metrics.sample(10))  # Show 10 random rows
