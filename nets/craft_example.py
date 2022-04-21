from keras import Sequential
from keras.layers import Dense, InputLayer
import tensorflow as tf
import numpy as np
import math
import tf2onnx


def paper_example(activation):
    model = Sequential()
    model.add(InputLayer((2,)))
    model.add(Dense(2, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(2, activation=None))

    model.set_weights([
        # First layer
        np.asarray([[1., 1.], [1., -1.]]),
        np.asarray([0., 0.]),
        # Second layer
        np.asarray([[1., 1.], [1., -1.]]),
        np.asarray([0., 0.]),
        # Final layer
        np.asarray([[1., 0.], [1., 1.]]),
        np.asarray([1., 0.]),
    ])
    return model


def onnx_paper_example(activation: str):
    model = paper_example(activation)
    return tf2onnx.convert.from_keras(model, output_path=f"paper_example_{activation}.onnx")


def test_relu_model():
    inputs = tf.convert_to_tensor([
        [0.5, 0.5],
        [0.5, 0.25],
        [0.25, 0.5]
    ])

    expected_outputs = np.asarray([
        [3., 1.],
        [2.5, 0.5],
        [2.5, 0.75]
    ])

    model = paper_example("ReLU")
    result = model(inputs).numpy()
    assert(np.allclose(result, expected_outputs))


def test_elu_model():
    inputs = tf.convert_to_tensor([
        [0.5, 0.5],
        [0.5, 0.25],
        [0.25, 0.5]
    ])

    expected_outputs = np.asarray([
        [3., 1.],
        [2.5, 0.5],
        [2.5, 0.75 - (math.exp(-.25) - 1)]
    ])

    model = paper_example("ELU")
    result = model(inputs).numpy()
    assert(np.allclose(result, expected_outputs))


if __name__ == "__main__":
    test_relu_model()
    test_elu_model()
    onnx_paper_example("ReLU")
    onnx_paper_example("ELU")
