from autograd import Value

x1 = Value(2.0)
x2 = Value(3.0)

w1 = Value(0.5)
w2 = Value(-1.0)

b = Value(0.5)

y = Value(1.0)

z = w1 * x1 + w2 * x2 + b
y_pred = z.tanh()

print(z.data)
print(y_pred.data)

loss = (y_pred - y) * (y_pred - y)
print(loss.data)

loss.backward()
print("w1: ", w1.grad)
print("w2: ", w2.grad)
print("b: ", b.grad)

learning_rate = 0.1

w1.data -= learning_rate * w1.grad
w2.data -= learning_rate * w2.grad
b.data -= learning_rate * b.grad

z = w1*x1 + w2*x2 + b
y_pred = z.tanh()

loss = (y_pred - y) * (y_pred - y)
print(loss.data)