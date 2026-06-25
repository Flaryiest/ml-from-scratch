## About
An autograd engine from scratch, following Karpathy's micrograd. A `Value` wraps a scalar and builds a computation graph as you do operations with these values.

The key idea at play is that you can determine the gradient or final effect of a Value at any point on the output, which is compared with the ground truth/desired outcome in order to get the loss. By slowly nudging the weights and biases towards a decreasing loss value, you end up with more accurate predictions.

In order to get accurate results/something moderately more trainable, the code forms a multilayer perceptron, which is composed of layers which are composed of neurons. Neurons have a fixed amount of inputs and corresponding initially random weights that get adjusted as gradient descent is performed. Additionally there is a random bias value.

## Key Notes to Remember
- grad must be zeroed before each backward pass
- topological sort = post-order DFS, then process the nodes in reverse
- each output node stores a `_backward` closure that knows how to push its gradient back to the inputs that produced it
- grad uses `+=` not `=`, because a node can feed into multiple others, so its gradient is the sum over all paths (this is the same reason grad must be zeroed before each pass)
- operator overloading (`__add__`, `__mul__`, etc.) is what lets you write `a * b + c` on your own Value objects  