# src/tfidf.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def map3(actual, preds):
    for i, p in enumerate(preds):
        if p == actual:
            return 1 / (i + 1)
    return 0


def run_tfidf(train):
    """
    Runs the TF-IDF baseline on the training set.

    Returns
    -------
    final_map3 : float
    tfidf_predictions : list
    """

    combined_text = (
        train["prompt"] + " " +
        train["A"] + " " +
        train["B"] + " " +
        train["C"] + " " +
        train["D"] + " " +
        train["E"]
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(combined_text)

    scores = []
    tfidf_predictions = []

    for _, row in train.iterrows():

        prompt_vec = vectorizer.transform([row["prompt"]])

        similarities = {}

        for option in ["A", "B", "C", "D", "E"]:

            option_vec = vectorizer.transform([row[option]])

            sim = cosine_similarity(
                prompt_vec,
                option_vec
            )[0][0]

            similarities[option] = sim

        ranked_options = sorted(
            similarities,
            key=similarities.get,
            reverse=True
        )

        top3 = ranked_options[:3]

        tfidf_predictions.append(top3)

        score = map3(
            row["answer"],
            top3
        )

        scores.append(score)

    final_map3 = sum(scores) / len(scores)

    return final_map3, tfidf_predictions
