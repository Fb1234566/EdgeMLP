#include "../../include/activation_functions/Linear.h"

double Linear::activate(double x)
{
    return x;
}

double Linear::derivative(double x)
{
    return 1.0;
}

std::string Linear::name()
{
    return "Linear";
}


