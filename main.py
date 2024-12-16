from datasets import load_dataset
import pandas as pd
from datetime import datetime
import logging
from metrics_calculator import calculate_daily_metrics


# Configure info-level logging and format for cleaner output
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# -------------------------------------------
# Data Loading and Preparation
# -------------------------------------------
def load_influencer_data(sample_size=1000):
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


# -------------------------------------------
# Main Execution
# -------------------------------------------
if __name__ == "__main__":
    # Step 1: Load data
    df_posts = load_influencer_data(sample_size=1000)

    # Step 2: Calculate daily metrics
    logging.info("Calculating daily influencer metrics...")
    daily_metrics = calculate_daily_metrics(df_posts)

    # Display a sample of daily metrics
    logging.info("Sample of Daily Metrics:")
    print(daily_metrics.sample(10))
