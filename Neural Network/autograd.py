import math


class Value:
    def __init__(self, data) -> None:
        self.data = data
        self.grad = 0.0
        self._prev = set()
        self._backward = lambda: None

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for v in reversed(topo):
            v._backward()

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        result = Value(self.data + other.data)
        result._prev.add(self)
        result._prev.add(other)

        def _backward():
            self.grad += result.grad
            other.grad += result.grad

        result._backward = _backward
        return result

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        result = Value(self.data * other.data)
        result._prev.add(self)
        result._prev.add(other)

        def _backward():
            self.grad += result.grad * other.data
            other.grad += result.grad * self.data

        result._backward = _backward
        return result

    def __rmul__(self, other):
        return self * other

    def exp(self):
        result = Value(math.exp(self.data))
        result._prev.add(self)

        def _backward():
            self.grad += result.grad * result.data

        result._backward = _backward
        return result

    def tanh(self):
        result = Value(math.tanh(self.data))
        result._prev.add(self)

        def _backward():
            self.grad += result.grad * (1 - result.data**2)

        result._backward = _backward
        return result
