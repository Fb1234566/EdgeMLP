#include "../../include/activation_functions/Relu.h"

double Relu::activate(const double x)
{
    if (x <= 0.0)
    {
        return 0;
    }
    return x;
}

double Relu::derivative(const double x)
{
    if (x <= 0.0)
    {
        return 0;
    }
    return 1;
}

std::string Relu::name()
{
    return "ReLU";
}
