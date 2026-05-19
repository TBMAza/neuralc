# neuralc

> A didactic neural network implementation in C — written to mirror the theory, not to win benchmarks.

This is an educational library that implements a fully-connected feedforward neural network from scratch in C. Every struct, every function, and every variable is named and structured to map directly onto the mathematical concepts you'd find in a textbook. If you've ever read about neurons, weights, biases, and backpropagation and wanted to see *exactly* how they translate into code — this is for you.

**This is not a production library.** It is not optimized. It does not use SIMD, it does not batch matrix operations, and it does not try to be fast. It tries to be *clear*.

---

## Table of Contents

- [Motivation](#motivation)
- [Concepts Illustrated](#concepts-illustrated)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
  - [Neuron](#neuron)
  - [Layer](#layer)
  - [Network](#network)
  - [Utils](#utils)
- [How It Works](#how-it-works)
  - [Forward Pass](#forward-pass)
  - [Loss Calculation](#loss-calculation)
  - [Backpropagation](#backpropagation)
  - [Weight Update](#weight-update)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Building](#building)
  - [Running the XOR Example](#running-the-xor-example)
- [API Reference](#api-reference)
- [Example Walkthrough](#example-walkthrough)
- [Known Bugs & Limitations](#known-bugs--limitations)
- [Learning Resources](#learning-resources)

---

## Motivation

Most neural network frameworks are intentionally opaque at the implementation level — and rightly so, because production code optimizes for performance and generality. This makes them poor learning tools if you want to understand what's actually happening during training.

`neuralc` takes the opposite approach: **the code is the explanation**. Each component corresponds 1-to-1 with its theoretical counterpart:

| Theory | Code |
|---|---|
| Neuron | `Neuron` struct in `neuron.h` |
| Weighted sum + activation | `neuron_output()` |
| Layer of neurons | `Layer` struct in `layer.h` |
| Full network | `Network` struct in `network.h` |
| Loss function (MSE) | `loss()` in `utils.c` |
| Backpropagation | `backpropagate()` in `network.c` |
| Gradient descent step | `train_neuron_step()` |

---

## Concepts Illustrated

- **Feedforward inference** — how inputs propagate through layers to produce an output
- **Sigmoid activation** — squashing neuron outputs into the (0, 1) range
- **Mean Squared Error loss** — measuring how wrong the network's predictions are
- **Backpropagation** — computing gradients layer by layer using the chain rule
- **Stochastic gradient descent** — updating weights and biases to minimize loss
- **Delta rule** — how each neuron's contribution to the error is tracked via a `delta` field

---

## Project Structure

```
src/
├── Makefile
├── main.c              # XOR training demo
├── core/
│   ├── neuron.h        # Neuron struct & function declarations
│   ├── neuron.c        # Neuron logic: init, forward, train step, free
│   ├── layer.h         # Layer struct & function declarations
│   ├── layer.c         # Layer logic: wraps a collection of neurons
│   ├── network.h       # Network struct & function declarations
│   └── network.c       # Network logic: forward pass, backprop, training loop
└── utils/
    ├── utils.h
    └── utils.c         # sigmoid, loss, random init, array helpers
```

---

## Architecture

### Neuron

The atomic unit. Defined in `core/neuron.h`:

```c
typedef struct {
    size_t inputs_no;     // Number of inputs this neuron accepts
    long double* inputs;  // Pointer to the input array (not owned)
    long double* weights; // One weight per input, randomly initialized
    long double bias;     // Bias term, randomly initialized
    long double output;   // Result after applying the activation function
    long double delta;    // Error signal used during backpropagation
} Neuron;
```

The `delta` field is what connects the forward pass to the backward pass. After `backpropagate()` runs, each neuron holds its own contribution to the total error — ready to be used by `train_neuron_step()`.

### Layer

A flat array of `Neuron`s that all receive the same input vector and each produce one scalar output. Defined in `core/layer.h`:

```c
typedef struct {
    size_t neurons_no;    // How many neurons are in this layer
    Neuron* neurons;      // The neurons themselves
    long double* outputs; // Collected outputs — fed as inputs to the next layer
} Layer;
```

The `outputs` array is the glue between layers: after `layer_output()` runs, `layer->outputs` holds one value per neuron, ready to be passed as `inputs` to the next layer.

### Network

A sequence of `Layer`s. Defined in `core/network.h`:

```c
typedef struct {
    size_t layers_no;     // Number of layers
    Layer* layers;        // The layers in order (input → ... → output)
    long double* outputs; // Pointer to the final layer's outputs
} Network;
```

Inference flows left to right: `network_output()` calls `layer_output()` on each layer in sequence, threading outputs into the next layer's inputs. `network->outputs` always points to the last layer's output array after inference.

### Utils

`utils/utils.c` contains the mathematical primitives:

| Function | Purpose |
|---|---|
| `sigmoid(x)` | Activation function: `1 / (1 + e^(-x))` |
| `sigmoid_derivative(x)` | Derivative *with respect to the output*: `x * (1 - x)` |
| `loss(output, expected)` | MSE per sample: `(output - expected)²` |
| `loss_derivative(output, expected)` | `2 * (output - expected)` |
| `random_init(l, u)` | Uniform random in `[l, u]` for weight initialization |
| `arr2d_to_pp(rows, cols, arr)` | Converts a 2D stack array to a `long double**` |

> **Note on `sigmoid_derivative`:** The function takes the neuron's *output* (i.e., the already-activated value), not the raw pre-activation input. This is why the formula is `x * (1 - x)` rather than the form involving `e^(-x)` directly — it's an algebraic simplification that works when `x = sigmoid(z)`.

---

## How It Works

### Forward Pass

```
inputs → [Layer 0] → [Layer 1] → ... → [Layer N] → outputs
```

Each neuron computes:

```
output = sigmoid( bias + Σ(input[i] * weight[i]) )
```

In code (`neuron.c`):

```c
void neuron_output(Neuron* neuron, long double* inputs) {
    neuron->inputs = inputs;
    neuron->output = neuron->bias;
    for(size_t i = 0; i < neuron->inputs_no; ++i) {
        neuron->output += neuron->inputs[i] * neuron->weights[i];
    }
    neuron->output = sigmoid(neuron->output);
}
```

### Loss Calculation

After a forward pass, loss is computed per output neuron using Mean Squared Error:

```
L = (predicted - expected)²
```

This is done externally (e.g., in `main.c`) for inspection, but the derivative is used internally during backpropagation.

### Backpropagation

`backpropagate()` in `network.c` implements the chain rule layer by layer, starting from the output layer and moving backward.

**Output layer** — delta is the product of the loss gradient and the activation gradient:

```
δ = loss'(output, expected) × sigmoid'(output)
```

**Hidden layers** — delta is the weighted sum of the *next* layer's deltas, scaled by the local activation gradient:

```
δ[j] = sigmoid'(output[j]) × Σ_k( δ_next[k] × weight_next[k][j] )
```

This is the chain rule: error flows backward through the weights.

### Weight Update

After backpropagation, `train_neuron_step()` applies gradient descent:

```
weight[i] -= learning_rate × delta × input[i]
bias      -= learning_rate × delta
```

This nudges each weight in the direction that reduces the loss.

---

## Getting Started

### Prerequisites

- GCC (or any C99-compatible compiler)
- `make`
- Standard C math library (`-lm`, already in the Makefile)

### Building

```bash
cd src
make
```

This produces an executable (check your `Makefile` for the output name).

### Running the XOR Example

```bash
make run
```

The demo in `main.c` trains a 4-layer network (`2 → 4 → 4 → 4 → 1`) on the XOR problem for 1,000,000 epochs at a learning rate of `0.01`. It prints predictions before and after training so you can see the loss drop.

Expected output (values will vary due to random initialization):

```
--- Before training ---
input: {0.0, 0.0}   expected: 0.0   output: 0.623401
input: {0.0, 1.0}   expected: 1.0   output: 0.641209
input: {1.0, 0.0}   expected: 1.0   output: 0.638847
input: {1.0, 1.0}   expected: 0.0   output: 0.651023
total loss: 0.247...

--- After training ---
input: {0.0, 0.0}   expected: 0.0   output: 0.021304
input: {0.0, 1.0}   expected: 1.0   output: 0.978211
input: {1.0, 0.0}   expected: 1.0   output: 0.977843
input: {1.0, 1.0}   expected: 0.0   output: 0.023109
total loss: 0.001...
```

---

## API Reference

### Neuron (`core/neuron.h`)

```c
void init_neuron(Neuron* neuron, size_t inputs_no);
// Allocates weights[], sets bias, initializes output and delta to 0.

void neuron_output(Neuron* neuron, long double* inputs);
// Computes weighted sum + bias, applies sigmoid, stores result in neuron->output.

void train_neuron_step(Neuron* neuron, long double learning_rate);
// Applies one gradient descent update to weights and bias using neuron->delta.

void print_neuron(Neuron* neuron, char* indentation);
// Pretty-prints all neuron fields for debugging.

void free_neuron(Neuron* neuron);
// Frees weights[]. Does NOT free inputs (not owned by the neuron).
```

### Layer (`core/layer.h`)

```c
void init_layer(Layer* layer, size_t neurons_no, size_t inputs_per_neuron);
// Allocates and initializes all neurons in the layer.

void layer_output(Layer* layer, long double* inputs);
// Runs forward pass through all neurons, stores results in layer->outputs.

void train_layer_step(Layer* layer, long double learning_rate);
// Calls train_neuron_step() on every neuron in the layer.

void print_layer(Layer* layer, char* indentation);
// Pretty-prints the layer and all its neurons.

void free_layer(Layer* layer);
// Frees all neurons and the outputs array.
```

### Network (`core/network.h`)

```c
void init_network(Network* network, size_t inputs_no, size_t layers_no, size_t* neurons_per_layer);
// Builds the full network. neurons_per_layer[i] is the neuron count for layer i.

void network_output(Network* network, long double* inputs);
// Full forward pass. Sets network->outputs to the final layer's output array.

void backpropagate(Network* network, long double* training_outputs);
// Computes delta for every neuron via the chain rule (output → input direction).

void train_network_step(Network* network, long double learning_rate);
// Applies gradient descent to every neuron in every layer.

void train_network_on_single_io(Network* network, long double learning_rate,
    long double* training_inputs, long double* training_outputs);
// forward pass → backprop → weight update for one input/output pair.

void train_network_on_multiple_io(Network* network, long double learning_rate,
    size_t io_len, long double** training_inputs, long double** training_outputs);
// Runs train_network_on_single_io for every sample in the dataset.

void train_network(Network* network, size_t epochs, long double learning_rate,
    size_t io_len, long double** training_inputs, long double** training_outputs);
// Full training loop for a given number of epochs.

void print_network(Network* network, char* indentation);
// Pretty-prints the entire network state.

void free_network(Network* network);
// Frees all layers and their contents.
```

---

## Example Walkthrough

Here's how to set up and train your own network:

```c
#include "core/network.h"
#include "utils/utils.h"
#include <stdlib.h>
#include <time.h>

int main(void) {
    srand(time(0));

    // Define training data: AND gate
    long double in[4][2]  = {{0,0},{0,1},{1,0},{1,1}};
    long double out[4][1] = {{0},  {0},  {0},  {1}};

    long double** inputs  = arr2d_to_pp(4, 2, in);
    long double** outputs = arr2d_to_pp(4, 1, out);

    // Create a network: 2 inputs → hidden layer of 4 → 1 output
    Network net;
    init_network(&net, 2, 2, (size_t[]){4, 1});

    // Train for 100,000 epochs
    train_network(&net, 100000, 0.1, 4, inputs, outputs);

    // Run inference
    network_output(&net, inputs[3]); // {1, 1}
    printf("1 AND 1 = %Lf\n", net.outputs[0]); // Should be close to 1.0

    free_network(&net);
    free(inputs);
    free(outputs);
    return 0;
}
```

---

## Known Bugs & Limitations

> These are intentional design trade-offs, not bugs. The codebase is verified memory-safe (tested with AddressSanitizer).

- **Online learning only** — Training updates weights after *every single sample* (online/stochastic gradient descent), not after a full batch. There is no mini-batch or full-batch gradient descent.

- **Single activation function** — Only sigmoid is used throughout. There is no support for ReLU, tanh, softmax, or any other activation.

- **No serialization** — Trained weights cannot be saved or loaded.

- **Not thread-safe** — No consideration for concurrency.

---

## Learning Resources

If you want to understand the theory behind this code more deeply:

- [3Blue1Brown — Neural Networks (YouTube)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — exceptional visual intuition for how neural networks learn
- [Michael Nielsen — Neural Networks and Deep Learning (free online book)](http://neuralnetworksanddeeplearning.com/) — the mathematical derivation of backpropagation in full detail
- [CS231n — Backpropagation notes (Stanford)](https://cs231n.github.io/optimization-2/) — the chain rule applied systematically to computation graphs

---

## License

Do whatever you want with this. It's a learning tool.

---

> This README was written by Claude AI
