#ifndef EDGEMLP_LINEAR_H
#define EDGEMLP_LINEAR_H

#include "Activation.h"

class Linear: public Activation
{
public:
    double activate(double x) override;
    double derivative(double x) override;
    std::string name() override;
};


#endif //EDGEMLP_LINEAR_H