#ifndef EDGEMLP_MATRIX_H
#define EDGEMLP_MATRIX_H

#include <iostream>
#include <vector>

class Matrix
{
private:
    int rows;
    int cols;
    std::vector<double> data;
public:
    Matrix(int rows, int cols);
    ~Matrix();
    double& operator()(int row, int col);
    double operator()(int row, int col) const;

    int getRows() const;
    int getCols() const;
    const double* getData() const;
    Matrix operator*(const Matrix& other);
    Matrix operator+(const Matrix& other);
    Matrix transpose();
    Matrix hadamardProduct(const Matrix& other);
    Matrix operator*(double scalar);
    Matrix operator+(double scalar);
    void randomize(const int min, const int max);
    void xavierInit();
};

std::ostream& operator<<(std::ostream& os, Matrix& matrix);
#endif //EDGEMLP_MATRIX_H