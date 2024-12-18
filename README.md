# ETL Pipeline for Apple-Related BlueSky Posts

This repository provides a Python-based ETL pipeline that extracts posts from a large dataset of BlueSky social media posts, filters them for Apple-related content, loads them into a SQLite database, computes daily engagement metrics, and categorizes authors into influencer types. The pipeline then displays the top 5 authors within each influencer category to flag potential new influencers for Apple products. 

---

## Prerequisites
- [Docker](https://docs.docker.com/get-docker/)

## Building the Docker Image
Run the following command to build the Docker image:
```bash
docker build -t bluesky_etl .
```

If you're using a machine with an ARM-based processor (e.g., Apple Silicon/M1/M2 Mac), you need to specify the --platform flag to emulate the amd64 architecture required by the build.
```bash
docker build -t --platform linux/amd64-t bluesky_etl .
```

## Running the Docker Image
Run the following command to run the Docker image:
```bash
docker run -t bluesky_etl .
```

If you're using a machine with an ARM-based processor (e.g., Apple Silicon/M1/M2 Mac), you need to specify the --platform flag to emulate the amd64 architecture required by the build.
```bash
docker run -t --platform linux/amd64-t bluesky_etl .
```

## Key Features

### 1. **Extraction**
- The pipeline streams data from the `alpindale/two-million-bluesky-posts` dataset hosted on **Hugging Face**.
- It processes posts in batches to prevent excessive memory usage.

### 2. **Transformation (Apple Keyword Filtering)**
- Posts are filtered by searching for **Apple-related keywords** (e.g., `iphone`, `macbook`, `apple watch`).
- Only posts containing these keywords proceed to the loading phase.

### 3. **Loading into SQLite**
- Filtered posts are inserted into a `posts` table within a local SQLite database (`posts.db`).
- Duplicate URIs are avoided by preloading existing URIs and filtering already-seen posts.

### 4. **Daily Metrics Calculation**
After loading Apple-related posts, the pipeline calculates daily metrics for each `(date, author)` pair:
- `post_count`: Number of posts by the author on that date.
- `replies_received`: Number of replies received by an author to any posts on that date.
- `matched_keywords`: Aggregated Apple-related keywords found in their posts.
- `engagement_ratio`: `replies_received / post_count`.
- `influencer_type`: Categorizes the author based on their posting and engagement patterns.

The pipeline **upserts** daily metrics into a `metrics` table using **ON CONFLICT** resolution to ensure the most up-to-date metrics.

### 5. **Top 5 Influencers by Category**
- Authors are grouped by their `influencer_type`, and the **top 5 authors** for each category are displayed based on their `post_count`.
- **Influential Creator:** High `post_count` and high `replies_received`.
- **Thought Leader:** Lower `post_count` but high `replies_received`.
- **Broadcaster:** High `post_count` but comparatively fewer `replies_received`.
- **General User:** Falls outside the above categories.

---

### Script Workflow
- Stream data from Hugging Face in batches.
- Filter Apple-related posts and insert them into the database.
- Compute daily metrics and upsert them into the metrics table.
- Display the top 5 authors in each influencer category.

### Incremental Updates
It is unclear if new data comes in monotonically, so the pipeline is not currently able to extract data incrementally. Each run processes the dataset from scratch. To optimize for repeated runs:
- Store last_processed_time to process only new data.
- Update filtering and loading logic to handle incremental updates efficiently.