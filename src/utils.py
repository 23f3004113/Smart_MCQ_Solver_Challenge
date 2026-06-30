def map3_score(true_answer, prediction):
    if true_answer == prediction[0]:
        return 1

    elif true_answer == prediction[1]:
        return 1 / 2

    elif true_answer == prediction[2]:
        return 1 / 3

    return 0
