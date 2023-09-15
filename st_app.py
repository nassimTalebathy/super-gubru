import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import asyncio
import datetime as dt
import time
import openai
import langchain

import src.bru_predictor as bru
import src.utils as utils

# Environment
load_dotenv(".env", override=True)
langchain.verbose = True
MOCK = os.environ.get("MOCK", "false") == "true"  # If it's not there, we're in prod

# Set page config
st.set_page_config(page_title="Super-GuBru", page_icon=":rugby:", layout="wide")
today = dt.datetime.today().date()

# Reset OPENAI key on startup
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = "abcdef" if MOCK else ""
if "min_date" not in st.session_state:
    # Set it to today
    st.session_state["min_date"] = today
if "auth_validated" not in st.session_state:
    st.session_state["auth_validated"] = False


#### Data ####
@st.cache_data
def load_fixtures_df() -> pd.DataFrame:
    print(f"Loading fixures from {st.session_state['min_date']} onwards...")
    df = pd.read_csv("rugby_world_cup_2023_fixtures.csv", parse_dates=["date"])
    # Timestamp
    df["time_str"] = (
        df.raw.str.replace("(noon,", "(12pm,", regex=False).str.extract(
            "\(([\dpm\.]{3,}), "
        )
        # .fillna('12pm')
    )
    df["hour"] = (
        df["time_str"]
        .str.extract("^(\d{1,2}).*[p\.]")
        .astype(pd.Int64Dtype())
        .fillna(12)
    )
    df["hour"] = df["hour"].where(df["hour"] >= 12, df["hour"] + 12).astype(str)
    df["minutes"] = df["time_str"].str.extract("^\d{1,2}\.(\d{2})").fillna("00")
    df["timestamp"] = pd.to_datetime(
        df["date"].dt.strftime("%Y-%m-%d") + " " + df["hour"] + ":" + df["minutes"],
        format="%Y-%m-%d %H:%M",
    )
    df.drop(columns=["time_str", "hour", "minutes"], inplace=True)
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
    df.sort_values(["timestamp", "home"], inplace=True)
    return df


def filter_df(tdf: pd.DataFrame) -> pd.DataFrame:
    return tdf[
        # Ignore knockout
        (tdf["pool"] != "KNOCKOUT")
        &
        # Ignore prior matches
        (tdf["date"].dt.date >= st.session_state["min_date"])
    ]


def line_break():
    return st.write("  \n")


def text_area(label: str, value: str, height: int = None, label_visibility="hidden"):
    if height is None:
        height = round(len(value) / 5)
    return st.text_area(
        label=label,
        value=value,
        height=height,
        label_visibility=label_visibility,
        disabled=True,
    )


fixtures_df = load_fixtures_df()
fixtures_options = filter_df(fixtures_df).index.to_list()

#### End data ####

################# Start of app #################
st.title("Super-Gubru")
st.subheader("Predicting SuperBru results since '23")

with st.sidebar:
    min_prediction_date = st.date_input(
        label="Minimum date (best to leave as is)",
        min_value=fixtures_df["date"].dt.date.min(),
        max_value=fixtures_df["date"].dt.date.max(),
        key="min_date",
    )
    openai_api_key = st.text_input(
        label="OpenAI API key",
        value=st.session_state["OPENAI_API_KEY"],
        type="password",
        key="OPENAI_API_KEY",
    )


# Select a fixture
selected_fixture = st.selectbox(
    label="Select a fixure", options=[None] + fixtures_options
)

if (
    openai_api_key is not None
    and openai_api_key != ""
    and selected_fixture is not None
    and selected_fixture != ""
):
    # API Key
    openai.api_key = openai_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key
    # Create the gubru
    gubru = bru.get_gubru(mock=MOCK)
    # Extract home, away
    row = fixtures_df.loc[selected_fixture]
    home, away = row["home"], row["away"]

    # Predict button
    predict_button = st.button("I'm feeling lucky...")

    # Left side: vanilla ChatGPT
    # Right side: our prediction
    vanilla_chatgpt_container, enhanced_chatgpt_container = st.columns([0.35, 0.65])

    if predict_button:
        # Vanilla ChatGPT block
        with vanilla_chatgpt_container:
            vanilla_chatgpt_container.text("What would ChatGPT answer?")
            # Initialise convo
            system_msg = "You are a helpful assistant"
            query = (
                "Predict the winner and points difference for this 2023 Rugby World Cup fixture: "
                + selected_fixture
            )
            # Write messages to app
            # vanilla_chatgpt_container.chat_message("assistant", avatar="🤖").write(
            #     system_msg
            # )
            vanilla_chatgpt_container.chat_message("human", avatar="😃").write(query)
            if MOCK:
                prediction = "As an AI assistant, I cannot predict future events. The outcome of a rugby match depends on various factors, including the performance of the teams on the day of the match. Therefore, it is not possible for me to accurately predict the winner or the points difference for the New Zealand v Namibia fixture in the 2023 Rugby World Cup."
            else:
                prediction = utils.vanilla_chatgpt_response(
                    query=query, system_msg=system_msg
                )
            vanilla_chatgpt_container.chat_message("assistant", avatar="🤖").write(
                prediction
            )

        # Our block
        with enhanced_chatgpt_container:
            enhanced_chatgpt_container.text("What would Super-GuBru answer?")
            with st.status("Looking at my crystal ball...", expanded=False) as status:
                status.write("Traversing the stratosphere for sources... (Search)")
                search_results = gubru.search_queries(
                    home=home, away=away, num_results=4
                )
                status.write("Pulling from the sources... (Scrape)")
                docs = asyncio.run(
                    gubru.get_docs_from_search_results(
                        search_results=search_results, first_chunk_only=True
                    )
                )
                status.write("Extraction process... (summarise)")
                if not MOCK:
                    time.sleep(10)
                status.write("Applying higher order thinking... (initial prediction)")
                if not MOCK:
                    time.sleep(8)
                status.write(
                    "Reviewing higher order thinking with even higher order thinking... (review + finalise)"
                )
                if not MOCK:
                    time.sleep(5)
                full_output = gubru.run(home=home, away=away, docs=docs)
                status.write("Back to earth")

            # Write out final result
            res: bru.MatchPrediction = full_output[gubru.reviewer_chain.output_key]
            st.markdown(f"### :red[Predicted Winner: {res.winner}]")
            st.markdown(f"#### :blue[Winning margin: {res.points_margin}]")
            text_area(
                "Reasoning", value=res.reasoning, label_visibility="visible", height=150
            )

        # Full output
        with st.expander("Want to see inside the ball?", expanded=False):
            # Convert sources into a list
            st.subheader("Sources used")
            st.write("\n".join(["- " + el["link"] for el in search_results]))
            line_break()
            # Summary
            st.subheader("Summary of information found")
            text_area(
                gubru.map_reduce_chain.output_key,
                value=full_output[gubru.map_reduce_chain.output_key],
            )
            line_break()
            # Thinking
            st.subheader("Initial thoughts")
            text_area(
                gubru.match_predictor_chain.output_key,
                value=full_output[gubru.match_predictor_chain.output_key],
            )
            line_break()
            # Review
            st.subheader("Final thoughts")
            # text_area(gubru.reviewer_chain.output_key, value=full_output[gubru.reviewer_chain.output_key])
            st.json(full_output[gubru.reviewer_chain.output_key].dict())
        # Appendix
        with st.expander("Appendix of scraped documents", expanded=False):
            for doc in full_output["scraped_docs"]:
                source = doc.metadata["source"]
                text_area(
                    "Content for " + source,
                    value="  \n".join(doc.page_content.split("\\n")),
                    height=400,
                    label_visibility="visible",
                )
