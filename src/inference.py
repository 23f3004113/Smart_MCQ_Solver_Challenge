import pandas as pd


def create_submission(predictions):

    submission = pd.DataFrame(
        {
            "prediction":predictions
        }
    )

    submission.to_csv(
        "../outputs/submission.csv",
        index=False
    )


if __name__=="__main__":

    preds=[
        "A B C"
    ]

    create_submission(preds)
