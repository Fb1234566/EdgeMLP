#include <iostream>
#include "Matrix.h"
#include "MLP.h"
#include "activation_functions/Sigmoid.h"
#include "activation_functions/Linear.h"
#include <vector>
#include <memory>

void printBanner() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║                                                  ║\n";
    std::cout << "║  ███████╗██████╗  ██████╗ ███████╗               ║\n";
    std::cout << "║  ██╔════╝██╔══██╗██╔════╝ ██╔════╝               ║\n";
    std::cout << "║  █████╗  ██║  ██║██║  ███╗█████╗                 ║\n";
    std::cout << "║  ██╔══╝  ██║  ██║██║   ██║██╔══╝                 ║\n";
    std::cout << "║  ███████╗██████╔╝╚██████╔╝███████╗               ║\n";
    std::cout << "║  ╚══════╝╚═════╝  ╚═════╝ ╚══════╝               ║\n";
    std::cout << "║                                                  ║\n";
    std::cout << "║  ███╗   ███╗██╗     ██████╗                      ║\n";
    std::cout << "║  ████╗ ████║██║     ██╔══██╗                     ║\n";
    std::cout << "║  ██╔████╔██║██║     ██████╔╝                     ║\n";
    std::cout << "║  ██║╚██╔╝██║██║     ██╔═══╝                      ║\n";
    std::cout << "║  ██║ ╚═╝ ██║███████╗██║                          ║\n";
    std::cout << "║  ╚═╝     ╚═╝╚══════╝╚═╝                          ║\n";
    std::cout << "║                                                  ║\n";
    std::cout << "║  Train MLPs in C++, deploy to Edge devices       ║\n";
    std::cout << "║                                                  ║\n";
    std::cout << "║  Version: 0.1.0                                  ║\n";
    std::cout << "║  Author: Fb1234566                               ║\n";
    std::cout << "║                                                  ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

int main()
{
    printBanner();

    std::vector<int> layer_sizes = {2, 3, 1};
    auto sigmoid = std::make_shared<Sigmoid>();
    auto linear = std::make_shared<Linear>();
    std::vector<std::shared_ptr<Activation>> activations = {sigmoid, linear};

    MLP mlp(layer_sizes, activations);

    std::cout << "MLP Architecture:" << std::endl;
    std::cout << mlp << std::endl;

    return 0;
}
