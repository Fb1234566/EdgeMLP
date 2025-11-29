#ifndef EDGEMLP_TANH_H
#define EDGEMLP_TANH_H

#include "Activation.h"

class Tanh: public Activation
{
public:
    double activate(double x) override;
    double derivative(double x) override;
    std::string name() override;
};


#endif //EDGEMLP_TANH_H