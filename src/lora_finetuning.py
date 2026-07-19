import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)


# -------------------------------
# Label Encoding
# -------------------------------

LABEL_MAP = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
}


# -------------------------------
# Load Model and Tokenizer
# -------------------------------

MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

base_model = AutoModelForMultipleChoice.from_pretrained(
    MODEL_NAME
)


# -------------------------------
# Apply LoRA
# -------------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)

lora_model = get_peft_model(
    base_model,
    lora_config,
)


# -------------------------------
# Dataset Preparation
# -------------------------------

def prepare_dataset(dataframe, max_length=64):

    dataset_items = []

    for _, row in dataframe.iterrows():

        choices = [
            f"{row['prompt']} [SEP] {row['A']}",
            f"{row['prompt']} [SEP] {row['B']}",
            f"{row['prompt']} [SEP] {row['C']}",
            f"{row['prompt']} [SEP] {row['D']}",
            f"{row['prompt']} [SEP] {row['E']}",
        ]

        tokens = tokenizer(
            choices,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        dataset_items.append(
            {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                "labels": LABEL_MAP[row["answer"]],
            }
        )

    return Dataset.from_list(dataset_items)


# -------------------------------
# Trainer Creation
# -------------------------------

def get_trainer(train_dataset):

    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        max_steps=4,
        logging_steps=1,
        report_to="none",
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DefaultDataCollator(),
    )

    return trainer


# -------------------------------
# Count Trainable Parameters
# -------------------------------

def count_trainable_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# -------------------------------
# Inference
# -------------------------------

def predict_probabilities(prompt, options):

    tokens = tokenizer(
        [
            f"{prompt} [SEP] {option}"
            for option in options
        ],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    device = next(lora_model.parameters()).device

    input_ids = tokens["input_ids"].unsqueeze(0).to(device)
    attention_mask = tokens["attention_mask"].unsqueeze(0).to(device)

    lora_model.eval()

    with torch.no_grad():

        outputs = lora_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )

    return probabilities.squeeze().cpu().tolist()
