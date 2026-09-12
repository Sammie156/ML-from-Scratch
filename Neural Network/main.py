from autograd import Value
from layer import Layer


layer = Layer(2, 3)
x = [Value(2.0), Value(3.0)]

outputs = layer(x)

print(outputs)
print(len(outputs))
print(len(layer.parameters()))