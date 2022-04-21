import onnx
import copy


def relu_to_elu(model: onnx.ModelProto):
    model = copy.deepcopy(model)
    for node in model.graph.node:
        if node.op_type == 'Relu':
            node.op_type = 'Elu'
            node.name = node.name.replace('Relu', 'Elu')
    return model


def saved_relu_to_elu(loadpath: str, savepath: str):
    model = onnx.load_model(loadpath)
    model = relu_to_elu(model)
    onnx.checker.check_model(model)
    onnx.save_model(model, savepath)


if __name__ == "__main__":
    saved_relu_to_elu("mnist_relu_3_50.onnx", "mnist_elu_3_50.onnx")
