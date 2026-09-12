from layer import Layer
from autograd import Value
from neuron import Neuron


class MLP:
    def __init__(self, n_inputs, n_outputs):
        self.layers = []

        for n in n_outputs:
            layer = Layer(n_inputs, n)
            self.layers.append(layer)
            n_inputs = n

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        parameter = []

        for layer in self.layers:
            for p in layer.parameters():
                parameter.append(p)

        return parameter
