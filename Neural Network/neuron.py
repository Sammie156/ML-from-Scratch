import random
from autograd import Value


class Neuron:
    def __init__(self, n_inputs) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        result = Value(0.0)
        for i in range(len(x)):
            result += self.w[i] * x[i]

        result += self.b
        result = result.tanh()

        return result
