import pandas as pd
import numpy as np
import logging

# Configure info-level logging and format for cleaner output
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def add_date_column(df):
    """
    Add a 'date' column extracted from the 'created_at' timestamp.

    Args:
        df (pd.DataFrame): DataFrame containing the 'created_at' column.

    Returns:
        pd.DataFrame: Updated DataFrame with a 'date' column.
    """
    df["created_at"] = pd.to_datetime(
        df["created_at"], errors="coerce", utc=True
    )

    # Drop rows where created_at is invalid
    invalid_count = df["created_at"].isna().sum()
    if invalid_count > 0:
        logging.warning(
            f"Dropping {invalid_count} rows with invalid 'created_at' values."
        )
        df = df.dropna(subset=["created_at"])

    # Ensure all remaining values are datetime
    if not pd.api.types.is_datetime64_any_dtype(df["created_at"]):
        raise ValueError(
            "'created_at' column contains non-datetime values even after conversion."
        )

    # Debugging: Inspect a sample of the column
    logging.debug(
        "Sample 'created_at' values after conversion: %s",
        df["created_at"].head(),
    )

    # Add the date column
    df = df.copy()  # Add this line to ensure df is a copy
    df.loc[:, "date"] = df["created_at"].dt.date
    logging.info("Successfully added 'date' column.")
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


import numpy as np
import pandas as pd
import logging


def calculate_daily_metrics(df):
    """
    Calculate daily metrics for posts, aggregating by date and author.

    Args:
        df (pd.DataFrame): Input DataFrame containing posts with 'date', 'author', and 'matched_keywords'.

    Returns:
        pd.DataFrame: Aggregated metrics per day and author with influencer categorization.
    """
    logging.info(
        "Calculating daily metrics and aggregating by date and author..."
    )

    # Ensure 'date' column exists
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["created_at"]).dt.date

    # Define a function to aggregate and deduplicate keywords
    def aggregate_keywords(keywords):
        flat_list = ", ".join(filter(None, keywords)).split(", ")
        return ", ".join(sorted(set(flat_list)))

    # Aggregate metrics by date and author
    daily_metrics = df.groupby(["date", "author"], as_index=False).agg(
        post_count=("uri", "count"),
        replies_received=("reply_to", "count"),
        matched_keywords=("matched_keywords", aggregate_keywords),
    )

    # Add engagement ratio
    daily_metrics["engagement_ratio"] = (
        daily_metrics["replies_received"] / daily_metrics["post_count"]
    )
    daily_metrics.fillna({"engagement_ratio": 0}, inplace=True)

    # Add influencer type categorization
    logging.info("Categorizing influencer types...")
    conditions = [
        (daily_metrics["post_count"] > daily_metrics["post_count"].median())
        & (
            daily_metrics["replies_received"]
            > daily_metrics["replies_received"].median()
        ),
        (daily_metrics["post_count"] <= daily_metrics["post_count"].median())
        & (
            daily_metrics["replies_received"]
            > daily_metrics["replies_received"].median()
        ),
        (daily_metrics["post_count"] > daily_metrics["post_count"].median())
        & (
            daily_metrics["replies_received"]
            <= daily_metrics["replies_received"].median()
        ),
    ]
    categories = ["Influential Creator", "Thought Leader", "Broadcaster"]
    daily_metrics["influencer_type"] = np.select(
        conditions, categories, default="General User"
    )

    logging.info("Daily metrics successfully calculated.")
    return daily_metrics
