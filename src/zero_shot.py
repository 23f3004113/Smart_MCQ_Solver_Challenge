from transformers import pipeline

classifier = pipeline("zero-shot-classification")

prompt = train_dataset[1]["prompt"]

candidate_labels = [
    train_dataset[1]["A"],
    train_dataset[1]["B"],
    train_dataset[1]["C"]
]

result_softmax = classifier(
    prompt,
    candidate_labels=candidate_labels
)

result_sigmoid = classifier(
    prompt,
    candidate_labels=candidate_labels,
    multi_label=True
)

sum_softmax = sum(result_softmax["scores"])
sum_sigmoid = sum(result_sigmoid["scores"])

difference = abs(sum_softmax - sum_sigmoid)

print(difference)
