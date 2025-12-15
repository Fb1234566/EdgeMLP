#ifndef EDGEMLP_ACTIVATION_H
#define EDGEMLP_ACTIVATION_H

#include "Matrix.h"
class Activation
{
public:
    virtual ~Activation() = default;
    virtual double activate(double x) = 0;
    virtual double derivative(double x) = 0;
    Matrix forward(const Matrix& m);
    Matrix backward(const Matrix& upstreamGradient, const Matrix& activationOutput);
    virtual std::string name() = 0;
};

#endif //EDGEMLP_ACTIVATION_H