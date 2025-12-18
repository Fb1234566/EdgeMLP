#include "../include/MLP.h"

MLP::MLP(const std::vector<int>& sizes, const std::vector<std::shared_ptr<Activation>>& activation_funcs) : layer_size(sizes), activations(activation_funcs)
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
        int n_in = sizes[i];
        int n_out = sizes[i+1];

        Matrix w(n_out, n_in);
        w.heInit(); // He initialization
        weights.push_back(w);

        Matrix b(n_out, 1);
        biases.push_back(b);
    }
}

std::ostream& operator<<(std::ostream& os, const MLP m)
{
    for (size_t i = 0; i < m.layer_size.size(); ++i) {
        std::cout << "  Layer " << i << ": " << m.layer_size[i] << " neurons";
        if (i < m.layer_size.size() - 1) {
            std::cout << ", Activation: " << m.activations[i]->name();
        }
        std::cout << std::endl;
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


