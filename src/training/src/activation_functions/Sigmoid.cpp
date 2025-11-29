#include "../../include/activation_functions/Sigmoid.h"
#include <cmath>

double Sigmoid::activate(const double x)
{
    return std::exp(x) / (1+exp(x));
}

double Sigmoid::derivative(const double x)
{
    return activate(x) * (1-activate(x));
}

std::string Sigmoid::name()
{
    return "Sigmoid";
}

