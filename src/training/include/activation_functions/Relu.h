#ifndef EDGEMLP_RELU_H
#define EDGEMLP_RELU_H

#include "../Activation.h"

class Relu: public Activation
{
public:
    double activate(double x) override;
    double derivative(double x) override;
    std::string name() override;
};


#endif //EDGEMLP_RELU_H