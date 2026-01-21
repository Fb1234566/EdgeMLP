#include <gtest/gtest.h>
#include "../include/MLP.h"
#include "../include/Matrix.h"
#include "../include/activation_functions/Sigmoid.h"
#include "../include/activation_functions/Linear.h"
#include <vector>
#include <memory>

// Test to verify MLP construction and architecture
TEST(MLPTest, ConstructorAndArchitecture) {
    std::vector<int> layer_sizes = {2, 3, 1};
    auto sigmoid = std::make_shared<Sigmoid>();
    std::vector<std::shared_ptr<Activation>> activations = {sigmoid, sigmoid};

    MLP mlp(layer_sizes, activations);

    // Check architecture
    const auto& weights = mlp.weights;
    const auto& biases = mlp.biases;

    // Expected number of weight matrices and bias vectors
    EXPECT_EQ(weights.size(), 2);
    EXPECT_EQ(biases.size(), 2);

    // Check dimensions of weights
    // Layer 1: 3 neurons, 2 inputs -> 3x2 weight matrix
    EXPECT_EQ(weights[0].getRows(), 3);
    EXPECT_EQ(weights[0].getCols(), 2);

    // Layer 2: 1 neuron, 3 inputs -> 1x3 weight matrix
    EXPECT_EQ(weights[1].getRows(), 1);
    EXPECT_EQ(weights[1].getCols(), 3);

    // Check dimensions of biases
    // Layer 1: 3 neurons -> 3x1 bias vector
    EXPECT_EQ(biases[0].getRows(), 3);
    EXPECT_EQ(biases[0].getCols(), 1);

    // Layer 2: 1 neuron -> 1x1 bias vector
    EXPECT_EQ(biases[1].getRows(), 1);
    EXPECT_EQ(biases[1].getCols(), 1);
}

// Test the forward pass with a simple 2-layer network and known weights
TEST(MLPTest, ForwardPassSimple) {
    // Define a 2-layer network: 2 inputs, 2 hidden neurons, 1 output
    std::vector<int> layer_sizes = {2, 2, 1};
    // Use Linear activation for simplicity to check calculations
    auto linear = std::make_shared<Linear>();
    std::vector<std::shared_ptr<Activation>> activations = {linear, linear};

    MLP mlp(layer_sizes, activations);

    // Manually set weights and biases for predictable output
    auto& weights = mlp.weights;
    auto& biases = mlp.biases;

    // Layer 1 weights (2x2)
    weights[0](0, 0) = 0.1; weights[0](0, 1) = 0.2;
    weights[0](1, 0) = 0.3; weights[0](1, 1) = 0.4;

    // Layer 1 biases (2x1)
    biases[0](0, 0) = 0.1;
    biases[0](1, 0) = 0.2;

    // Layer 2 weights (1x2)
    weights[1](0, 0) = 0.5; weights[1](0, 1) = 0.6;

    // Layer 2 biases (1x1)
    biases[1](0, 0) = -0.1;

    // Input matrix (2x1)
    Matrix input(2, 1);
    input(0, 0) = 2.0; // Input 1
    input(1, 0) = 3.0; // Input 2

    // --- Manual Calculation ---
    // Hidden Layer (z1 = W1 * input + b1)
    // z1_0 = (0.1 * 2.0) + (0.2 * 3.0) + 0.1 = 0.2 + 0.6 + 0.1 = 0.9
    // z1_1 = (0.3 * 2.0) + (0.4 * 3.0) + 0.2 = 0.6 + 1.2 + 0.2 = 2.0
    // a1 = linear(z1) = z1

    // Output Layer (z2 = W2 * a1 + b2)
    // z2_0 = (0.5 * 0.9) + (0.6 * 2.0) - 0.1 = 0.45 + 1.2 - 0.1 = 1.55
    // a2 = linear(z2) = z2

    // Perform forward pass
    Matrix output = mlp.forward(input);

    // Check output dimensions
    EXPECT_EQ(output.getRows(), 1);
    EXPECT_EQ(output.getCols(), 1);

    // Check final output value
    EXPECT_NEAR(output(0, 0), 1.55, 1e-9);
}

TEST(MLPTest, BackwardPassSimpleLinear) {
    std::vector<int> layer_sizes = {2, 2, 1};
    auto linear = std::make_shared<Linear>();
    std::vector<std::shared_ptr<Activation>> activations = {linear, linear};
    MLP mlp(layer_sizes, activations);

    // Imposta pesi e bias noti
    auto& weights = mlp.weights;
    auto& biases = mlp.biases;
    weights[0](0, 0) = 0.1; weights[0](0, 1) = 0.2;
    weights[0](1, 0) = 0.3; weights[0](1, 1) = 0.4;
    biases[0](0, 0) = 0.1; biases[0](1, 0) = 0.2;
    weights[1](0, 0) = 0.5; weights[1](0, 1) = 0.6;
    biases[1](0, 0) = -0.1;

    // Input e output target
    Matrix input(2, 1);
    input(0, 0) = 2.0;
    input(1, 0) = 3.0;
    Matrix target(1, 1);
    target(0, 0) = 2.0;

    // Salva i valori originali
    double w1_before = weights[0](0, 0);
    double b2_before = biases[1](0, 0);

    // Esegui retropropagazione
    mlp.backpropagate(input, target);

    // Dopo la retropropagazione, i pesi e bias devono essere cambiati
    EXPECT_NE(weights[0](0, 0), w1_before);
    EXPECT_NE(biases[1](0, 0), b2_before);
}

// Test la retropropagazione con attivazione sigmoid
TEST(MLPTest, BackwardPassSimpleSigmoid) {
    std::vector<int> layer_sizes = {2, 2, 1};
    auto sigmoid = std::make_shared<Sigmoid>();
    std::vector<std::shared_ptr<Activation>> activations = {sigmoid, sigmoid};
    MLP mlp(layer_sizes, activations);

    // Inizializza pesi e bias
    auto& weights = mlp.weights;
    auto& biases = mlp.biases;
    weights[0](0, 0) = 0.2; weights[0](0, 1) = -0.3;
    weights[0](1, 0) = 0.4; weights[0](1, 1) = 0.1;
    biases[0](0, 0) = 0.0; biases[0](1, 0) = 0.0;
    weights[1](0, 0) = -0.5; weights[1](0, 1) = 0.2;
    biases[1](0, 0) = 0.1;

    // Input e output target
    Matrix input(2, 1);
    input(0, 0) = 0.5;
    input(1, 0) = -1.5;
    Matrix target(1, 1);
    target(0, 0) = 0.7;

    // Salva i valori originali
    double w2_before = weights[1](0, 0);
    double b1_before = biases[0](0, 0);

    // Esegui retropropagazione
    mlp.backpropagate(input, target);

    // Dopo la retropropagazione, i pesi e bias devono essere cambiati
    EXPECT_NE(weights[1](0, 0), w2_before);
    EXPECT_NE(biases[0](0, 0), b1_before);
}
