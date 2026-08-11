import pandas as pd


OPTIONS = ["A","B","C","D","E"]


def create_combined_text(df):

    texts=[]

    for _,row in df.iterrows():

        text = str(row["prompt"])

        for option in OPTIONS:
            text += " " + str(row[option])

        texts.append(text)

    return texts



def encode_labels(df):

    mapping = {
        "A":0,
        "B":1,
        "C":2,
        "D":3,
        "E":4
    }

    df=df.copy()

    df["label"] = (
        df["answer"]
        .map(mapping)
    )

    return df
