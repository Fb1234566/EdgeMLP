#include "Matrix.h"

#include <stdexcept>
#include <algorithm>
#include <random>

Matrix::Matrix(const int rows, const int cols) : rows(rows), cols(cols), data(rows * cols, 0.0)
{
}

int Matrix::getRows() const
{
    return rows;
}

int Matrix::getCols() const
{
    return cols;
}

double& Matrix::operator()(const int row, const int col)
{
    return data[row * cols + col];
}

double Matrix::operator()(const int row, const int col) const
{
    return data[row * cols + col];
}

const double* Matrix::getData() const
{
    return data.data();
}

Matrix::~Matrix()
{
}

Matrix Matrix::operator*(const Matrix& other)
{
    if (cols != other.rows)
    {
        throw std::invalid_argument(
            "Cannot multiply matrices with incompatible dimensions " + std::to_string(cols) + " and " +
            std::to_string(other.rows));
    }

    Matrix result(rows, other.cols);
#pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < other.cols; j++)
        {
            double sum = 0;
            for (int k = 0; k < cols; k++)
            {
                sum += (*this)(i, k) * other(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

Matrix Matrix::operator+(const Matrix& other)
{
    if (rows != other.rows || cols != other.cols)
    {
        throw std::invalid_argument(
            "Cannot add matrices with incompatible dimensions" + std::to_string(cols) + " and " + std::to_string(
                other.rows));
    }

    Matrix result(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result(i, j) = (*this)(i, j) + other(i, j);
        }
    }

    return result;
}

Matrix Matrix::transpose()
{
    Matrix result(cols, rows);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result(j, i) = (*this)(i, j);
        }
    }

    return result;
}

Matrix Matrix::hadamardProduct(const Matrix& other)
{
    Matrix result(rows, other.cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < other.cols; j++)
        {
            result(i, j) = (*this)(i, j) * other(i, j);
        }
    }

    return result;
}

Matrix Matrix::operator*(const double scalar)
{
    Matrix result(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result(i, j) = (*this)(i, j) * scalar;
        }
    }
    return result;
}

Matrix Matrix::operator+(const double scalar)
{
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result(i, j) = (*this)(i, j) + scalar;
        }
    }
    return result;
}

void Matrix::randomize(const double min, const double max)
{
    std::default_random_engine eng;
    std::uniform_real_distribution<double> distribution(min, max);
    std::for_each(data.begin(), data.end(), [distribution, eng](double& elem) mutable
    {
        elem = distribution(eng);
    });
}

void Matrix::xavierInit()
{
    const double fanIn = rows;
    const double fanOut = cols;

    const double sigma = std::sqrt(6.0 / (fanIn + fanOut));
    randomize(-sigma, sigma);
}

void Matrix::heInit()
{
    const double fanIn = rows;

    const double stdDeviation = std::sqrt(2 / fanIn);

    std::default_random_engine eng;
    std::normal_distribution<double> distribution(0, stdDeviation);
    std::for_each(data.begin(), data.end(), [distribution, eng](double& elem) mutable
    {
        elem = distribution(eng);
    });
}

double Matrix::sum() const
{
    return std::accumulate(data.begin(), data.end(), 0.0);
}

double Matrix::mean() const
{
    if (data.empty())
        return 0.0;
    return sum()/static_cast<double>(data.size());
}

Matrix Matrix::sumRows() const
{
    Matrix result(rows, 1);

    for (int i = 0; i<rows; i++)
    {
        for (int j = 0; j<cols; j++)
        {
            result(i, 0) += (*this)(i, j);
        }
    }

    return result;
}

Matrix& Matrix::operator=(const Matrix& m)
{
    Matrix tmp(m);
    std::swap(rows, tmp.rows);
    std::swap(cols, tmp.cols);
    std::swap(data, tmp.data);
    return *this;
}

Matrix Matrix::map(const std::function<double(double)>& func) const
{
    Matrix result(*this);
    std::transform(result.data.begin(), result.data.end(), result.data.begin(), func);
    return result;
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix)
{
    os << "Matrix [" << matrix.getRows() << "x" << matrix.getCols() << "]:\n";
    for (int i = 0; i < matrix.getRows(); ++i)
    {
        os << "[ ";
        for (int j = 0; j < matrix.getCols(); ++j)
        {
            os << matrix(i, j);
            if (j < matrix.getRows() - 1)
                os << ", ";
        }
        os << " ]\n";
    }
    os << std::endl;
    return os;
}
