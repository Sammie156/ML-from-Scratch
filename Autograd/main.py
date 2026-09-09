from autograd import Value

x1 = Value(2.0)
x2 = Value(3.0)

w1 = Value(0.5)
w2 = Value(-1.0)

b = Value(0.5)

y = Value(1.0)

learning_rate = 0.1

# for step in range(20):
z = w1*x1 + w2*x2 + b
y_pred = z.tanh()

loss = (y_pred - y) * (y_pred - y)

w1.zero_grad()
w2.zero_grad()
b.zero_grad()

loss.backward()

w1.data -= learning_rate * w1.grad
w2.data -= learning_rate * w2.grad
b.data -= learning_rate * b.grad

print(w1.grad)

# if step % 2 == 0:
#     print(step, "loss = ", loss.data, "w1 = ", w1.data, "w2 = ", w2.data, "b = ", b.data)