# ⚽ ScoutIQ — Football Player Recommendation System

## Project Overview

ScoutIQ is a football scouting and player recommendation application built using Python, machine learning, and Streamlit.

The project analyzes football player statistics and recommends players with similar playing profiles using cosine similarity.

ScoutIQ provides an interactive web application where users can search for players, discover statistically similar footballers, filter recommendations by position, and compare player performance.

## Dataset

The project uses publicly available football event data from StatsBomb.

- **Source:** StatsBomb Open Data
- **Competition:** German Bundesliga
- **Season:** 2023/24
- **Matches analyzed:** 34
- **Players in the dataset:** 373
- **Players eligible for recommendations:** 82
- **Minimum playing time:** 180 minutes

The available matches primarily feature Bayer Leverkusen and their opponents rather than the entire Bundesliga schedule.

## Technologies Used

- **Python** — Main programming language
- **Pandas** — Data processing and analysis
- **NumPy** — Numerical operations
- **Scikit-learn** — Feature standardization and cosine similarity
- **Streamlit** — Interactive web application
- **Matplotlib** — Player comparison visualizations in Jupyter
- **StatsBombPy** — Football event data collection
- **Jupyter Notebook** — Data exploration and model development

## Features

### 1. Player Recommendation Engine

ScoutIQ recommends players based on statistical similarities in their performances.

The recommendation system uses six performance metrics:

- Passing accuracy (%)
- Passes per 90 minutes
- Interceptions per 90 minutes
- Ball recoveries per 90 minutes
- Shots per 90 minutes
- Dribbles per 90 minutes

These metrics are standardized using `StandardScaler`, and cosine similarity is used to calculate the statistical resemblance between players.

### 2. Position-Based Filtering

Users can filter recommendations using three options:

- **Exact Position:** Recommend players who play the same specific position.
- **Same Position Group:** Recommend players who play similar roles, such as fullbacks and wingbacks.
- **All Positions:** Recommend players regardless of position.

### 3. Minimum Similarity Filtering

Users can adjust the minimum cosine similarity score to control which recommendations are displayed.

### 4. Interactive Web Application

ScoutIQ includes a Streamlit interface that allows users to:

- Search and select football players.
- View player information, including team, position, and approximate minutes played.
- Choose the number of recommendations.
- Filter recommendations by position.
- Adjust the minimum similarity threshold.
- Compare player performance statistics.
- Download recommendations as CSV files.

### 5. Player Performance Visualization

ScoutIQ provides interactive comparison charts showing standardized performance metrics for a selected player and recommended alternatives.

The Jupyter Notebook also contains player comparison visualizations created using Matplotlib.

## Project Structure

```text
ScoutIQ/
│
├── app.py
├── requirements.txt
├── README.md
├── ScoutIQ.ipynb
│
├── src/
│   ├── __init__.py
│   ├── scraper.py
│   └── recommender.py
│
└── data/
    ├── scoutiq_all_players.csv
    ├── scoutiq_eligible_players.csv
    ├── scoutiq_similarity_matrix.csv
    └── scoutiq_sample_recommendations.csv
```

**Main files:**

- `app.py` — Streamlit web application.
- `src/recommender.py` — Player similarity calculations and recommendation logic.
- `src/scraper.py` — StatsBomb data collection.
- `ScoutIQ.ipynb` — Data exploration, statistical analysis, and model development.
- `requirements.txt` — Python dependencies.
- `data/` — Exported player statistics and recommendation data.

## Installation and Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ScoutIQ.git
cd ScoutIQ
```

Replace `YOUR_USERNAME` with the actual GitHub username once the project is published.

### Step 2 — Install Dependencies

Make sure Python is installed, then run:

```bash
python -m pip install -r requirements.txt
```

### Step 3 — Prepare the Dataset

Make sure `scoutiq_eligible_players.csv` is located inside the `data/` directory.

This file contains the player statistics used by the recommendation engine.

### Step 4 — Run ScoutIQ

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open in your browser, typically at:

`http://localhost:8501`

## How It Works

ScoutIQ follows these steps:

1. **Data Collection:** Retrieve Bundesliga match events using StatsBombPy.
2. **Data Processing:** Combine event data from 34 matches into a single dataset.
3. **Feature Engineering:** Calculate passing, shooting, dribbling, and defensive statistics.
4. **Playing Time:** Estimate minutes played and calculate performance metrics per 90 minutes.
5. **Player Filtering:** Select players with at least 180 minutes of recorded playing time.
6. **Feature Standardization:** Standardize the six selected performance metrics using `StandardScaler`.
7. **Similarity Calculation:** Calculate cosine similarity between player feature vectors.
8. **Recommendations:** Rank similar players and apply the selected position and similarity filters.
9. **Visualization:** Display recommendations and player performance comparisons in Streamlit.

## Example Results

Using Alejandro Grimaldo García as the selected player

URL: https://scoutiqmh.streamlit.app/
