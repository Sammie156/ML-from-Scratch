from autograd import Value

x = Value(2.0)
y = Value(3.0)

z = x + y
q = z * x
r = q.tanh()

r.grad = 1.0
r.backward()

print(x.grad)
print(y.grad)

for v in r._prev:
    print(v.data)