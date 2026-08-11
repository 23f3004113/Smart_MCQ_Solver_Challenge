from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf():

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1,3),
        max_features=100000,
        min_df=2,
        sublinear_tf=True
    )

    return vectorizer
