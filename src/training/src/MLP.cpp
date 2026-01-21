#include "../include/MLP.h"

MLP::MLP(const std::vector<int>& sizes, const std::vector<std::shared_ptr<Activation>>& activation_funcs, double learning_rate) : layer_size(sizes), activations(activation_funcs), learning_rate(learning_rate)
{
    if (sizes.size() < 2)
    {
        throw std::invalid_argument("MLP must have at least 2 layers");
    }

    if  (sizes.size() - 1 != activation_funcs.size())
    {
        throw std::invalid_argument("The number of activation functions must be equal to the number of layers minus one");
    }

    for (size_t i{}; i < sizes.size() - 1; i++)
    {
        const int n_in = sizes[i];
        const int n_out = sizes[i+1];

        Matrix w(n_out, n_in);
        w.heInit(); // He initialization
        weights.push_back(w);

        Matrix b(n_out, 1);
        biases.push_back(b);
    }
}

std::ostream& operator<<(std::ostream& os, const MLP& m)
{
    for (size_t i = 0; i < m.layer_size.size(); ++i) {
        os << "  Layer " << i << ": " << m.layer_size[i] << " neurons";
        if (i < m.layer_size.size() - 1) {
            os << ", Activation: " << m.activations[i]->name();
        }
        os << std::endl;
    }
    return os;
}

Matrix MLP::forward(const Matrix& input)
{
    if (input.getRows() != layer_size[0] || input.getCols() != 1) {
        throw std::invalid_argument("Input matrix dimensions do not match the input layer size.");
    }

    z_values.clear();
    a_values.clear();

    Matrix current_a = input;
    a_values.push_back(current_a);


    for (size_t i {}; i < weights.size(); i++)
    {
        const Matrix& w = weights[i];
        const Matrix& b = biases[i];
        const auto& activation = activations[i];

        Matrix z = (w * current_a) + b;
        z_values.push_back(z);

        current_a = activation->forward(z);
        a_values.push_back(current_a);
    }

    return current_a;
}

void MLP::backpropagate(const Matrix& input, const Matrix& output)
{
    forward(input);

    std::vector<Matrix> nabla_b, nabla_w;
    for (const auto& b : biases) nabla_b.emplace_back(b.getRows(), b.getCols());
    for (const auto& w : weights) nabla_w.emplace_back(w.getRows(), w.getCols());

    // 1. Compute delta output
    Matrix delta = a_values.back() - output;
    delta = activations.back()->backward(delta, z_values.back());
    nabla_b.back() = delta;
    nabla_w.back() = delta * a_values[a_values.size() - 2].transpose();

    // 2. Propagation in the hidden layers
    for (int l = static_cast<int>(weights.size()) - 2; l >= 0; --l) {
        Matrix wT_delta = weights[l + 1].transpose() * delta;
        delta = activations[l]->backward(wT_delta, z_values[l]);
        nabla_b[l] = delta;
        nabla_w[l] = delta * a_values[l].transpose();
    }

    // 3. Update parameters
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = weights[i] - nabla_w[i] * learning_rate;
        biases[i] = biases[i] - nabla_b[i] * learning_rate;
    }
}

