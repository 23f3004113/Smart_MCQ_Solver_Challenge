from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice
)



MODEL_NAME="bert-base-uncased"



def load_bert():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )


    model = AutoModelForMultipleChoice.from_pretrained(
        MODEL_NAME
    )


    return tokenizer,model
