# import dependencies
import numpy as np
import pandas as pd
import os
import csv
import re
from html import unescape
from textblob import TextBlob
import altair as alt
import seaborn as sns
from matplotlib import pyplot as plt
import streamlit as st

# specify data's date
date = "02/27/2022"

# define data file path
data_path = "./data_2.0/UkraineCombinedTweetsDeduped_FEB27.csv"


# DATA PRE-PROCESSING HELPER FUNCTIONS
# abstract cleaning of tweets with functions
def remove_urls(x):  # --> remove urls
    cleaned_string = re.sub(
        r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b", "", str(x), flags=re.MULTILINE
    )
    return cleaned_string


def deEmojify(x):  # --> remove emoticons
    regrex_pattern = re.compile(
        pattern="["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+",
        flags=re.UNICODE,
    )
    return regrex_pattern.sub(r"", x)


def unify_whitespaces(x):  # --> unify whitespaces
    cleaned_string = re.sub(" +", " ", x)
    return cleaned_string


def remove_symbols(x):  # --> remove unwanted symbols and preserve sentence structure
    cleaned_string = re.sub(r"[^a-zA-Z0-9?!.,]+", " ", x)
    return cleaned_string


# create function to get polarity of TextBlob Model
def getPolarity(text):
    return TextBlob(text).sentiment.polarity


# extract sentiment from polarity scores
def getAnalysis(score):
    # negative sentiment
    if score < 0:
        return "negative"
    # neutral sentiment
    elif score == 0:
        return "neutral"
    # positive sentiment
    else:
        return "positive"


# CACHING INGESTION AND CLEANING OF DATA
# put code we only want to run once in a function
@st.cache
def get_and_label_data(file_path):
    of = pd.read_csv(file_path, index_col=0, encoding="utf-8")
    # subset english tweets
    df = of[of["language"] == "en"]

    # pre-preocess tweet column
    df["text"] = df["text"].str.lower()
    df["text"] = df["text"].apply(remove_urls)
    df["text"] = df["text"].apply(deEmojify)
    df["text"] = df["text"].apply(remove_symbols)
    df["text"] = df["text"].apply(unify_whitespaces)

    # create polarity column by applying
    df["polarity"] = df["text"].apply(getPolarity)

    # extract sentiment from polarity
    df["sentiment"] = df["polarity"].apply(getAnalysis)

    # convert the 'Date' column to datetime format
    df["hour"] = pd.to_datetime(df["tweetcreatedts"]).dt.hour

    # set Date column as index for easy subsetting
    df.set_index("hour")

    # reformat location values to upper case to eliminate inconsistencies
    df["location"] = df["location"].str.upper()

    return df


def main():
    """Streamlit App to Track Sentiment on Russia Ukrain War"""

    header = st.container()
    dataset = st.container()
    hourly_sentimennt_plot = st.container()
    select_country_section = st.container()

    # add style
    # st.markdown(
    #    """<style>.main{background-color: #F5F5F5;}</style>""", unsafe_allow_html=True
    # )

    with header:
        st.title("Russia Ukraine War Tweet Sentiment Analysis Tracker")
        st.caption("Team 3: Tego, Vicki, Haoliang, Godwin & John")
        st.text("Track Sentiment ")

    with dataset:
        st.header(
            "Daily Tweets on War from Kaggle with Sentiments Labeled by Machine Learning Algorithm"
        )
        st.text("First Few Rows of Data")
        df = get_and_label_data(data_path)
        st.write(df.head())
        st.write("The total number of tweets in data is:", len(df["text"]))

        st.subheader("Daily Counts of Sentiments from The World Over")
        sent_valcounts = pd.DataFrame(df["sentiment"].value_counts())
        st.bar_chart(sent_valcounts, use_container_width=True)

    with hourly_sentimennt_plot:
        st.header("Hourly Grouped Distribution of Daily Sentiments")
        fig_1, ax_1 = plt.subplots(figsize=(24, 11))
        sns.countplot(x="hour", data=df, hue="sentiment")
        plt.title(f"Aggregate Counts of Sentiments on {date}")
        plt.xlabel("Hour")
        plt.ylabel("Count")
        st.pyplot(fig_1)

    with select_country_section:
        st.header(
            f"Time to Explore the Sentiment of the People in a Given Location on {date}"
        )

        # to place widgets in columns, refer to column variables below
        location = df.location.unique().tolist()
        # get input from user
        option = st.selectbox("What location would you like to explore?", location)

        st.write("You selected:", option)
        df_selected = df[df["location"] == option]

        st.write(
            "In this location the total number of tweets on the war is:",
            len(df_selected.location),
        )

        st.subheader("Daily Counts of Sentiments From: ", option)
        sent_loc_valcounts = pd.DataFrame(df_selected["sentiment"].value_counts())
        st.bar_chart(sent_loc_valcounts)


if __name__ == "__main__":
    main()
