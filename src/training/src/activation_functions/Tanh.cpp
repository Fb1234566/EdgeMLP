#include "../../include/activation_functions/Tanh.h"
#include <cmath>

double Tanh::activate(const double x)
{
    return std::tanh(x);
}

double Tanh::derivative(const double x)
{
    const double y = std::tanh(x);
    return 1.0 - (y*y);
}

std::string Tanh::name()
{
    return "Hyperbolic tangent";
}


