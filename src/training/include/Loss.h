#ifndef EDGEMLP_LOSS_H
#define EDGEMLP_LOSS_H

#include "Matrix.h"

class Loss
{
public:
    virtual ~Loss() = default;
    virtual double calculate(const Matrix& output, const Matrix& target) const = 0;
    virtual Matrix derivative(const Matrix& output, const Matrix& target) const = 0;
};

#endif //EDGEMLP_LOSS_H