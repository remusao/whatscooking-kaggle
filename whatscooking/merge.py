

from __future__ import print_function


with open("output", "rb") as prediction_file:
    predictions = prediction_file.read().split()

with open("to_predict.txt", "rb") as input_data:
    with open("to_predict_labeled.txt", "wb") as output:
        for prediction, line in zip(predictions, input_data):
            line = str(prediction) + line[1:]
            output.write(line)
