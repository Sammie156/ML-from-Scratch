# Autograd

A basic autograd engine made with Python to understand how the algorithm works behind the scenes. Currently only addition, multiplication, exponential and tanh operations are supported.

## Working of Autograd

Autograd is basically an automatic differentiation engine. Whenever an operation is performed between two nodes, each holding a value, the autograd engine automatically stores the result and the cumulative derivative of each individual node up until that point. This helps in **Backpropagation**, when dealing with neural networks. Pretty interesting.
