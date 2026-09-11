import random
from autograd import Value


class Neuron:
    def __init__(self, n_inputs) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def parameters(self):
        parameter = []

        for w in self.w:
            parameter.append(w)

        parameter.append(self.b)

        return parameter

    def __call__(self, x):
        result = Value(0.0)

        for wi, xi in zip(self.w, x):
            result += wi * xi

        result += self.b
        result = result.tanh()

        return result
