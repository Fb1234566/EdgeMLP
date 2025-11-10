#include <iostream>
#include "Matrix.h"

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
    return 0;
}
