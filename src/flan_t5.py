from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-small"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small"
)

row = train_dataset[0]

prompt = (
    f"Question: {row['prompt']}. "
    f"Is the correct answer A: {row['A']} or B: {row['B']}? "
    f"Answer with just the letter A or B."
)

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

outputs = model.generate(
    **inputs,
    max_new_tokens=5
)

generated = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print(generated)
