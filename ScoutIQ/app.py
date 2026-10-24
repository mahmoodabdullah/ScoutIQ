from pathlib import Path

import pandas as pd
import streamlit as st

from src.recommender import FEATURES, ScoutIQRecommender

st.set_page_config(page_title="ScoutIQ", page_icon="⚽", layout="wide")
st.title("⚽ ScoutIQ")
st.caption("Football player similarity scouting | StatsBomb Bundesliga 2023/24 sample")

DATA_PATH = Path(__file__).resolve().parent / "data" / "scoutiq_eligible_players.csv"


@st.cache_resource

def load_model():
    return ScoutIQRecommender(DATA_PATH)


try:
    model = load_model()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.info("In Jupyter, run: eligible_players.to_csv('scoutiq_eligible_players.csv', index=False) and move the CSV into ScoutIQ/data/.")
    st.stop()

players = model.players
st.sidebar.header("Search settings")
selected = st.sidebar.selectbox("Select player", sorted(players["player"].tolist()))
top_n = st.sidebar.slider("Number of recommendations", 1, 10, 5)
filter_mode = st.sidebar.radio("Position filter", ["Same position group", "Exact position", "All positions"])
minimum = st.sidebar.slider("Minimum cosine similarity", -1.0, 1.0, 0.0, 0.05)

profile = players.set_index("player").loc[selected]
st.subheader(selected)
a, b, c = st.columns(3)
a.metric("Team", str(profile["team"]))
b.metric("Position", str(profile["position"]))
c.metric("Minutes (approx.)", f"{profile['minutes_played']:,.0f}")

results = model.recommend_players(
    selected, top_n=top_n,
    same_position=filter_mode == "Exact position",
    same_group=filter_mode == "Same position group",
    min_similarity=minimum,
)
st.subheader("Similar players")
if results.empty:
    st.warning("No players meet these filters. Try lowering the similarity threshold or selecting all positions.")
else:
    st.dataframe(
        results.rename(columns={
            "player": "Player", "similarity_score": "Cosine similarity",
            "team": "Team", "position": "Position",
            "position_group": "Position group", "minutes_played": "Minutes",
        }),
        use_container_width=True, hide_index=True,
    )
    names = [selected] + results["player"].head(3).tolist()
    raw, standardized = model.comparison(names)
    st.subheader("Performance comparison")
    st.caption("Standardized feature values (z-scores) across the eligible player pool; higher is not always better.")
    st.bar_chart(standardized.T)
    with st.expander("Raw player statistics"):
        st.dataframe(raw, use_container_width=True)
    st.download_button(
        "Download recommendations (CSV)",
        results.to_csv(index=False).encode("utf-8"),
        file_name="scoutiq_recommendations.csv", mime="text/csv",
    )

st.divider()
st.caption(
    "Prototype limitations: available games center on Bayer Leverkusen; opposition players have fewer observed matches. "
    "Playing time is approximate. Similarity is not a measure of player quality or transfer suitability."
)
