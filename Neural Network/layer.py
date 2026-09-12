from neuron import Neuron


class Layer:
    def __init__(self, n_inputs, n_outputs) -> None:
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def parameters(self):
        parameter = []

        for neuron in self.neurons:
            for p in neuron.parameters():
                parameter.append(p)

        return parameter

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]
