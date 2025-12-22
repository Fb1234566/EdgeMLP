#ifndef EDGEMLP_MSE_H
#define EDGEMLP_MSE_H

#include "../Loss.h"

class MSE: public Loss
{
public:
    double calculate(const Matrix& output, const Matrix& target) const override;
    Matrix derivative(const Matrix& output, const Matrix& target) const override;
};

#endif //EDGEMLP_MSE_H