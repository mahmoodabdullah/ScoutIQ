"""Download StatsBomb open event data for a competition/season.

Usage: python -m src.scraper --competition-id 9 --season-id 281
This is optional: the Streamlit app reads the CSV exported from the notebook.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from statsbombpy import sb


def download_events(competition_id=9, season_id=281, output_path="data/combined_events.csv", delay=0.2):
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    match_ids = matches["match_id"].dropna().unique()
    frames = []
    for index, match_id in enumerate(match_ids, start=1):
        print(f"Downloading match {index}/{len(match_ids)}: {match_id}")
        events = sb.events(match_id=int(match_id))
        events["match_id"] = int(match_id)
        frames.append(events)
        if delay:
            time.sleep(delay)
    if not frames:
        raise ValueError("No match events were returned.")
    combined = pd.concat(frames, ignore_index=True)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    print(f"Saved {len(combined):,} events from {len(match_ids)} matches to {path}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download StatsBomb open match events")
    parser.add_argument("--competition-id", type=int, default=9)
    parser.add_argument("--season-id", type=int, default=281)
    parser.add_argument("--output", default="data/combined_events.csv")
    args = parser.parse_args()
    download_events(args.competition_id, args.season_id, args.output)
