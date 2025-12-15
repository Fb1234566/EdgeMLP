#ifndef EDGEMLP_SIGMOID_H
#define EDGEMLP_SIGMOID_H

#include "../Activation.h"

class Sigmoid: public Activation
{
public:
    double activate(double x) override;
    double derivative(double x) override;
    std::string name() override;
};


#endif //EDGEMLP_SIGMOID_H