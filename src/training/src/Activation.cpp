#include "../include/Activation.h"

Matrix Activation::forward(const Matrix& m)
{
    Matrix res = m.map([this](const double x) { return this->activate(x); });
    return res;
}

Matrix Activation::backward(const Matrix& upstreamGradient, const Matrix& activationOutput)
{
    Matrix localGradient = activationOutput.map([this](const  double x){return this->derivative(x);});
    Matrix res = localGradient.hadamardProduct(upstreamGradient);
    return res;
}
