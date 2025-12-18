#ifndef EDGEMLP_MLP_H
#define EDGEMLP_MLP_H

#include <memory>
#include <vector>

#include "Activation.h"
#include "Matrix.h"

class MLP
{
public:
    MLP() = default;
    MLP(const std::vector<int>& sizes, const std::vector<std::shared_ptr<Activation>>& activations);
    friend std::ostream& operator<<(std::ostream& os, const MLP& m);
    Matrix forward(const Matrix& input);

    std::vector<Matrix> weights;
    std::vector<Matrix> biases;

private:
    std::vector<int> layer_size;
    std::vector<std::shared_ptr<Activation>> activations;
    std::vector<Matrix> z_values;
    std::vector<Matrix> a_values;
};

std::ostream& operator<<(std::ostream& os, const MLP& m);


#endif //EDGEMLP_MLP_H
