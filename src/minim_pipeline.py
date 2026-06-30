from sentence_transformers import SentenceTransformer, util
from utils import map3_score

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

labels = ["A", "B", "C", "D", "E"]

mini_scores = []
mini_predictions = []

for row in train_dataset:

    prompt_embedding = model.encode(
        row["prompt"],
        convert_to_tensor=True
    )

    similarities = []

    for option in labels:

        option_embedding = model.encode(
            row[option],
            convert_to_tensor=True
        )

        score = util.cos_sim(
            prompt_embedding,
            option_embedding
        ).item()

        similarities.append(score)

    ranking = sorted(
        zip(labels, similarities),
        key=lambda x: x[1],
        reverse=True
    )

    top3 = [x[0] for x in ranking[:3]]

    mini_predictions.append(top3)

    mini_scores.append(
        map3_score(
            row["answer"],
            top3
        )
    )
