from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "passing_accuracy", "passes_per90", "interceptions_per90",
    "ball_recoveries_per90", "shots_per90", "dribbles_per90",
]
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "scoutiq_eligible_players.csv"


class ScoutIQRecommender:
    def __init__(self, csv_path=DEFAULT_DATA):
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Player dataset not found: {path}. Export eligible_players from your notebook "
                "as scoutiq_eligible_players.csv into the data/ folder."
            )
        players = pd.read_csv(path)
        required = ["player", "team", "position", "position_group", "minutes_played"] + FEATURES
        missing = [col for col in required if col not in players.columns]
        if missing:
            raise ValueError(f"Missing required CSV columns: {missing}")
        players = players.dropna(subset=["player", "position_group"]).copy()
        players[FEATURES] = players[FEATURES].apply(pd.to_numeric, errors="coerce")
        players["minutes_played"] = pd.to_numeric(players["minutes_played"], errors="coerce")
        players = players.dropna(subset=FEATURES + ["minutes_played"])
        players = players[players["minutes_played"] >= 180]
        players = players.drop_duplicates(subset="player").reset_index(drop=True)
        if len(players) < 2:
            raise ValueError("At least two eligible players are required.")
        self.players = players
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(players[FEATURES])
        self.similarity_df = pd.DataFrame(
            cosine_similarity(scaled),
            index=players["player"], columns=players["player"],
        )

    def recommend_players(self, player_name, top_n=5, same_position=False,
                          same_group=False, min_similarity=0):
        if player_name not in self.similarity_df.columns:
            raise ValueError(f"Player not found: {player_name}")
        scores = self.similarity_df[player_name].drop(index=player_name)
        target = self.players.set_index("player").loc[player_name]
        indexed = self.players.set_index("player")
        if same_position:
            allowed = indexed.index[indexed["position"] == target["position"]]
            scores = scores[scores.index.isin(allowed)]
        elif same_group:
            allowed = indexed.index[indexed["position_group"] == target["position_group"]]
            scores = scores[scores.index.isin(allowed)]
        scores = scores[scores >= min_similarity].sort_values(ascending=False).head(top_n)
        result = scores.rename("similarity_score").reset_index().rename(columns={"player": "player"})
        return result.merge(
            self.players[["player", "team", "position", "position_group", "minutes_played"]],
            on="player", how="left",
        )

    def comparison(self, names):
        selected = self.players.set_index("player").loc[names, FEATURES]
        scaled = pd.DataFrame(
            self.scaler.transform(selected), index=selected.index, columns=FEATURES
        )
        return selected, scaled
