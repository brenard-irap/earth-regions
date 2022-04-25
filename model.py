import tensorflow.keras as keras
import pandas as pd


def run_prediction(x_input, index):
    model = keras.models.load_model("data/best_model.hdf5")

    y_prob = model.predict(x_input)
    y_classes = y_prob.argmax(axis=-1)

    y_classes = pd.DataFrame({'classes': y_classes})
    y_classes['Time'] = index

    return y_classes
