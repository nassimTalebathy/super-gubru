import streamlit as st
import pandas as pd
import src.bru_predictor as bru
from dotenv import load_dotenv
import os
import asyncio
import datetime as dt
import time
import openai


st.set_page_config(page_title="Super-GuBru", page_icon=":rugby:", layout="wide")
load_dotenv(".env", override=True)
# Reset OPENAI key on startup
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""


@st.cache_data
def load_fixtures_df() -> list[str]:
    print("Loading fixures...")
    df = pd.read_csv("rugby_world_cup_2023_fixtures.csv", parse_dates=["date"])
    df = df[
        # Ignore knockout
        (df["pool"] != "KNOCKOUT")
        &
        # Ignore prior matches
        (df["date"] >= dt.datetime.today())
    ]
    df["label"] = (
        df["home"]
        + " v "
        + df["away"]
        + " on "
        + df["date"].dt.strftime("%a %d %b")
        # + " at "
        # + df["venue"]
    )
    df.set_index("label", inplace=True)
    return df


fixtures_df = load_fixtures_df()
fixtures_options = fixtures_df.index.to_list()


st.title("Super-Gubru")
st.subheader("Predicting SuperBru results since '23")

with st.sidebar:
    openai_api_key = st.text_input(
        label="OPENAI API KEY",
        value=st.session_state["OPENAI_API_KEY"],
        type="password",
        key="OPENAI_API_KEY",
    )


selected_fixture = st.selectbox(
    label="Select a fixure", options=[None] + fixtures_options
)

if openai_api_key is not None and openai_api_key != "" and selected_fixture is not None:
    # API Key
    openai.api_key = openai_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key
    # Create the gubru
    gubru = bru.SuperGuBru()
    # Extract home, away
    row = fixtures_df.loc[selected_fixture]
    home, away = row["home"], row["away"]
    with st.status("Looking at my crystal ball...", expanded=False) as status:
        status.write("Traversing the stratosphere for sources... (Search)")
        search_results = gubru.search_queries(home=home, away=away, num_results=3)
        status.write("Pulling from the sources... (Scrape)")
        docs = asyncio.run(
            gubru.get_docs_from_search_results(
                search_results=search_results, first_chunk_only=True
            )
        )
        status.write("Extraction process... (summarise)")
        status.write("Applying higher order thinking... (initial prediction)")
        time.sleep(1)
        status.write(
            "Reviewing higher order thinking with even higher order thinking... (review + finalise)"
        )
        full_output = gubru.run(home=home, away=away, docs=docs)
        status.write("Back to earth")

    # Write out final result
    res: bru.MatchPrediction = full_output[gubru.reviewer_chain.output_key]
    c = st.container()
    c.markdown(f"### :red[Predicted Winner: {res.winner}]")
    c.markdown(f"#### :blue[Winning margin: {res.points_margin}]")
    c.write(f"Thoughts:\n{res.reasoning}")

    # Full output
    with st.expander("Want to see inside the ball?", expanded=False):
        st.write("Summary of information found")
        st.write(full_output[gubru.map_reduce_chain.output_key])
        st.write("Sources used")
        st.write([el["link"] for el in search_results])
        st.write("Initial thoughts")
        st.write(full_output[gubru.match_predictor_chain.output_key])
        st.write("Rest")
        st.json(full_output)
