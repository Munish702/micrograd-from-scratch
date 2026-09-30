# micrograd from scratch

A tiny autograd engine and neural network library built from the ground up in plain Python, to understand how backpropagation actually works under the hood.

Built by following Andrej Karpathy's lecture [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0).

## What's inside

Everything is in [`micrograd.ipynb`](micrograd.ipynb):

1. **Derivatives, intuitively**: measuring slopes numerically by nudging inputs.
2. **The `value` class**: wraps a single number and records the operations that produced it. Supports `+`, `-`, `*`, `/`, `**`, `exp` and `tanh`, and computes gradients for all of them with `backward()`.
3. **Computation graph visualisation** with Graphviz.
4. **Backprop by hand**: gradients of a small expression worked out with the chain rule and checked numerically.
5. **A single neuron**: backprop done manually node by node, then automatically, with matching results.
6. **Checking against PyTorch**: the same neuron in PyTorch gives identical gradients.
7. **A multi-layer perceptron**: `Neuron` → `Layer` → `MLP`, trained with gradient descent on a small dataset.

## How backprop works here

Each operation (like `a * b`) creates a new `value` and attaches a small `_backward` function that knows the local derivative of that operation. Calling `backward()` on the output:

1. sorts all nodes so each one comes after its inputs (topological sort),
2. sets the output's gradient to 1,
3. walks the graph in reverse, calling each node's `_backward` to pass the gradient to its inputs via the chain rule.

Gradients are accumulated with `+=`, so a value used in several places gets contributions from every path. For the same reason, gradients are reset to zero before every training step.

## Results

The MLP (3 inputs, two hidden layers of 4 neurons, 1 output, 41 parameters) learns the 4 training examples in 20 steps of gradient descent:

| Target | Prediction |
|-------:|-----------:|
|   1.0  |      0.836 |
|  -1.0  |     -0.906 |
|  -1.0  |     -0.807 |
|   1.0  |      0.867 |

The loss drops from about 5.2 to 0.09. Exact numbers depend on the random seed.

## Running it

```bash
pip install numpy matplotlib graphviz torch jupyter
```

The graph drawings also need the Graphviz program itself:

- macOS: `brew install graphviz`
- Ubuntu: `sudo apt install graphviz`
- Windows: install from [graphviz.org](https://graphviz.org/download/) and tick "Add Graphviz to the system PATH"

Then open `micrograd.ipynb` and choose **Restart & Run All**. PyTorch is only used in section 6 to verify the results.

## What I learned

- How the chain rule turns into a simple, local "pass the gradient back" rule at every node.
- Why gradients must be accumulated (`+=`) and reset to zero before each training step.
- How complex operations like `tanh` or division can be built from a few basic ones that already know their derivatives.
- That PyTorch's autograd does the same thing, just on tensors instead of single numbers.

## Credits

Based on [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy and his accompanying lecture.
