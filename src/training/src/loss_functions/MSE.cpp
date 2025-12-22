#include "../include/loss_functions/MSE.h"

double MSE::calculate(const Matrix& output, const Matrix& target) const
{
    Matrix error = output - target;
    const Matrix squared_error = error.hadamardProduct(error);
    return squared_error.mean();
}

Matrix MSE::derivative(const Matrix& output, const Matrix& target) const
{
    const int n = output.getRows() * output.getCols();

    if (n==0)
    {
        return Matrix(0, 0);
    }

    Matrix error = output - target;

    const double scalar = 2.0/n;

    return error * scalar;
}
