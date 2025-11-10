#ifndef EDGEMLP_MATRIX_H
#define EDGEMLP_MATRIX_H

#include <iostream>
#include <vector>
#include <functional>

class Matrix
{
private:
    int rows;
    int cols;
    std::vector<double> data;
public:
    Matrix(int rows, int cols);
    ~Matrix();
    Matrix(const Matrix& m)=default;
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
    void randomize(double min, double max);
    void xavierInit();
    void heInit();
    double sum() const;
    double mean() const;
    Matrix sumRows() const;
    Matrix map(const std::function<double(double)>& func) const;
    Matrix& operator=(const Matrix& m);
    void
};

std::ostream& operator<<(std::ostream& os, const Matrix& matrix);
#endif //EDGEMLP_MATRIX_H