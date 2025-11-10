#include <gtest/gtest.h>
#include "../include/Matrix.h"
#include <sstream>
#include <cmath>
#include <limits>

// Test to verify matrix construction and dimensions
TEST(MatrixTest, ConstructorAndDimensions) {
    const int rows = 3;
    const int cols = 4;
    const Matrix m(rows, cols);

    EXPECT_EQ(m.getRows(), rows);
    EXPECT_EQ(m.getCols(), cols);
}

// Test to verify element access and modification
TEST(MatrixTest, ElementAccess)
{
    Matrix m(2, 2);
    m(0, 0) = 1.0;
    m(0, 1) = 2.0;
    m(1, 0) = 3.0;
    m(1, 1) = 4.0;

    EXPECT_DOUBLE_EQ(m(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m(1, 1), 4.0);
}

// Test for const element access
TEST(MatrixTest, ConstElementAccess)
{
    Matrix m(2, 2);
    m(0, 0) = 5.0;
    m(1, 1) = 10.0;

    const Matrix& constM = m;
    EXPECT_DOUBLE_EQ(constM(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(constM(1, 1), 10.0);
}

// Test for getData method
TEST(MatrixTest, GetData)
{
    Matrix m(2, 2);
    m(0, 0) = 1.0;
    m(0, 1) = 2.0;
    m(1, 0) = 3.0;
    m(1, 1) = 4.0;

    const double* data = m.getData();
    EXPECT_DOUBLE_EQ(data[0], 1.0);
    EXPECT_DOUBLE_EQ(data[1], 2.0);
    EXPECT_DOUBLE_EQ(data[2], 3.0);
    EXPECT_DOUBLE_EQ(data[3], 4.0);
}

// Test for matrix multiplication
TEST(MatrixTest, MatrixMultiplication)
{
    Matrix m1(2, 3);
    m1(0, 0) = 1.0; m1(0, 1) = 2.0; m1(0, 2) = 3.0;
    m1(1, 0) = 4.0; m1(1, 1) = 5.0; m1(1, 2) = 6.0;

    Matrix m2(3, 2);
    m2(0, 0) = 7.0; m2(0, 1) = 8.0;
    m2(1, 0) = 9.0; m2(1, 1) = 10.0;
    m2(2, 0) = 11.0; m2(2, 1) = 12.0;

    Matrix result = m1 * m2;

    EXPECT_EQ(result.getRows(), 2);
    EXPECT_EQ(result.getCols(), 2);
    EXPECT_DOUBLE_EQ(result(0, 0), 58.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 64.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 139.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 154.0);
}

// Test for matrix multiplication with incompatible dimensions
TEST(MatrixTest, MatrixMultiplicationIncompatible)
{
    Matrix m1(2, 3);
    Matrix m2(2, 2);

    EXPECT_THROW(m1 * m2, std::invalid_argument);
}

// Test for matrix addition
TEST(MatrixTest, MatrixAddition)
{
    Matrix m1(2, 2);
    m1(0, 0) = 1.0; m1(0, 1) = 2.0;
    m1(1, 0) = 3.0; m1(1, 1) = 4.0;

    Matrix m2(2, 2);
    m2(0, 0) = 5.0; m2(0, 1) = 6.0;
    m2(1, 0) = 7.0; m2(1, 1) = 8.0;

    Matrix result = m1 + m2;

    EXPECT_DOUBLE_EQ(result(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 8.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 10.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 12.0);
}

// Test for matrix addition with incompatible dimensions
TEST(MatrixTest, MatrixAdditionIncompatible)
{
    Matrix m1(2, 2);
    Matrix m2(2, 3);

    EXPECT_THROW(m1 + m2, std::invalid_argument);
}

// Test for matrix transpose
TEST(MatrixTest, Transpose)
{
    Matrix m(2, 3);
    m(0, 0) = 1.0; m(0, 1) = 2.0; m(0, 2) = 3.0;
    m(1, 0) = 4.0; m(1, 1) = 5.0; m(1, 2) = 6.0;

    Matrix result = m.transpose();

    EXPECT_EQ(result.getRows(), 3);
    EXPECT_EQ(result.getCols(), 2);
    EXPECT_DOUBLE_EQ(result(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(result(2, 0), 3.0);
    EXPECT_DOUBLE_EQ(result(2, 1), 6.0);
}

// Test for Hadamard product (element-wise multiplication)
TEST(MatrixTest, HadamardProduct)
{
    Matrix m1(2, 2);
    m1(0, 0) = 1.0; m1(0, 1) = 2.0;
    m1(1, 0) = 3.0; m1(1, 1) = 4.0;

    Matrix m2(2, 2);
    m2(0, 0) = 5.0; m2(0, 1) = 6.0;
    m2(1, 0) = 7.0; m2(1, 1) = 8.0;

    Matrix result = m1.hadamardProduct(m2);

    EXPECT_DOUBLE_EQ(result(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 12.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 21.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 32.0);
}

// Test for scalar multiplication
TEST(MatrixTest, ScalarMultiplication)
{
    Matrix m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(1, 0) = 3.0; m(1, 1) = 4.0;

    Matrix result = m * 2.5;

    EXPECT_DOUBLE_EQ(result(0, 0), 2.5);
    EXPECT_DOUBLE_EQ(result(0, 1), 5.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 7.5);
    EXPECT_DOUBLE_EQ(result(1, 1), 10.0);
}

// Test for scalar addition
TEST(MatrixTest, ScalarAddition)
{
    Matrix m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(1, 0) = 3.0; m(1, 1) = 4.0;

    Matrix result = m + 5.0;

    EXPECT_DOUBLE_EQ(result(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 7.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 8.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 9.0);
}

// Test for randomize method
TEST(MatrixTest, Randomize)
{
    Matrix m(3, 3);
    m.randomize(-1.0, 1.0);

    bool allInRange = true;
    for (int i = 0; i < m.getRows(); i++)
    {
        for (int j = 0; j < m.getCols(); j++)
        {
            if (m(i, j) < -1.0 || m(i, j) > 1.0)
            {
                allInRange = false;
                break;
            }
        }
    }

    EXPECT_TRUE(allInRange);
}

// Test for Xavier initialization
TEST(MatrixTest, XavierInit)
{
    Matrix m(3, 3);
    m.xavierInit();

    bool hasNonZero = false;
    for (int i = 0; i < m.getRows(); i++)
    {
        for (int j = 0; j < m.getCols(); j++)
        {
            if (m(i, j) != 0.0)
            {
                hasNonZero = true;
                break;
            }
        }
    }

    EXPECT_TRUE(hasNonZero);
}

// Test for He initialization
TEST(MatrixTest, HeInit)
{
    Matrix m(3, 3);
    m.heInit();

    bool hasNonZero = false;
    for (int i = 0; i < m.getRows(); i++)
    {
        for (int j = 0; j < m.getCols(); j++)
        {
            if (m(i, j) != 0.0)
            {
                hasNonZero = true;
                break;
            }
        }
    }

    EXPECT_TRUE(hasNonZero);
}

// Test for sum of all elements
TEST(MatrixTest, Sum)
{
    Matrix m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(1, 0) = 3.0; m(1, 1) = 4.0;

    double sum = m.sum();
    EXPECT_DOUBLE_EQ(sum, 10.0);
}

// Test for mean of all elements
TEST(MatrixTest, Mean)
{
    Matrix m(2, 2);
    m(0, 0) = 2.0; m(0, 1) = 4.0;
    m(1, 0) = 6.0; m(1, 1) = 8.0;

    double mean = m.mean();
    EXPECT_DOUBLE_EQ(mean, 5.0);
}

// Test for sum of rows
TEST(MatrixTest, SumRows)
{
    Matrix m(2, 3);
    m(0, 0) = 1.0; m(0, 1) = 2.0; m(0, 2) = 3.0;
    m(1, 0) = 4.0; m(1, 1) = 5.0; m(1, 2) = 6.0;

    Matrix result = m.sumRows();

    EXPECT_EQ(result.getRows(), 2);
    EXPECT_EQ(result.getCols(), 1);
    EXPECT_DOUBLE_EQ(result(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 15.0);
}

// Test for map function
TEST(MatrixTest, Map)
{
    Matrix m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(1, 0) = 3.0; m(1, 1) = 4.0;

    Matrix result = m.map([](double x) { return x * 2.0; });

    EXPECT_DOUBLE_EQ(result(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 6.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 8.0);
}

// Test for applyFunction method
TEST(MatrixTest, ApplyFunction)
{
    Matrix m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(1, 0) = 3.0; m(1, 1) = 4.0;

    m.applyFunction([](double x) { return x + 10.0; });

    EXPECT_DOUBLE_EQ(m(0, 0), 11.0);
    EXPECT_DOUBLE_EQ(m(0, 1), 12.0);
    EXPECT_DOUBLE_EQ(m(1, 0), 13.0);
    EXPECT_DOUBLE_EQ(m(1, 1), 14.0);
}

// Test for copy constructor
TEST(MatrixTest, CopyConstructor)
{
    Matrix m1(2, 2);
    m1(0, 0) = 1.0; m1(0, 1) = 2.0;
    m1(1, 0) = 3.0; m1(1, 1) = 4.0;

    Matrix m2(m1);

    EXPECT_EQ(m2.getRows(), 2);
    EXPECT_EQ(m2.getCols(), 2);
    EXPECT_DOUBLE_EQ(m2(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m2(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m2(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m2(1, 1), 4.0);
}

// Test for assignment operator
TEST(MatrixTest, AssignmentOperator)
{
    Matrix m1(2, 2);
    m1(0, 0) = 1.0; m1(0, 1) = 2.0;
    m1(1, 0) = 3.0; m1(1, 1) = 4.0;

    Matrix m2(3, 3);
    m2 = m1;

    EXPECT_EQ(m2.getRows(), 2);
    EXPECT_EQ(m2.getCols(), 2);
    EXPECT_DOUBLE_EQ(m2(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m2(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m2(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m2(1, 1), 4.0);
}

// Test for output stream operator
TEST(MatrixTest, OutputStreamOperator)
{
    Matrix m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(1, 0) = 3.0; m(1, 1) = 4.0;

    std::ostringstream oss;
    oss << m;

    std::string output = oss.str();
    EXPECT_TRUE(output.find("Matrix") != std::string::npos);
    EXPECT_TRUE(output.find("[2x2]") != std::string::npos);
}

// ===== EDGE CASE TESTS =====

// Test with 1x1 matrix
TEST(MatrixEdgeCaseTest, SingleElementMatrix)
{
    Matrix m(1, 1);
    m(0, 0) = 42.0;

    EXPECT_EQ(m.getRows(), 1);
    EXPECT_EQ(m.getCols(), 1);
    EXPECT_DOUBLE_EQ(m(0, 0), 42.0);
    EXPECT_DOUBLE_EQ(m.sum(), 42.0);
    EXPECT_DOUBLE_EQ(m.mean(), 42.0);
}

// Test multiplication with 1x1 matrix
TEST(MatrixEdgeCaseTest, SingleElementMultiplication)
{
    Matrix m1(1, 1);
    m1(0, 0) = 5.0;

    Matrix m2(1, 1);
    m2(0, 0) = 3.0;

    Matrix result = m1 * m2;
    EXPECT_DOUBLE_EQ(result(0, 0), 15.0);
}

// Test transpose of 1xN row vector
TEST(MatrixEdgeCaseTest, RowVectorTranspose)
{
    Matrix m(1, 5);
    for (int i = 0; i < 5; i++)
        m(0, i) = i + 1.0;

    Matrix result = m.transpose();
    EXPECT_EQ(result.getRows(), 5);
    EXPECT_EQ(result.getCols(), 1);
    for (int i = 0; i < 5; i++)
        EXPECT_DOUBLE_EQ(result(i, 0), i + 1.0);
}

// Test transpose of Nx1 column vector
TEST(MatrixEdgeCaseTest, ColumnVectorTranspose)
{
    Matrix m(5, 1);
    for (int i = 0; i < 5; i++)
        m(i, 0) = i + 1.0;

    Matrix result = m.transpose();
    EXPECT_EQ(result.getRows(), 1);
    EXPECT_EQ(result.getCols(), 5);
    for (int i = 0; i < 5; i++)
        EXPECT_DOUBLE_EQ(result(0, i), i + 1.0);
}

// Test with all zero values
TEST(MatrixEdgeCaseTest, AllZerosMatrix)
{
    Matrix m(3, 3);

    EXPECT_DOUBLE_EQ(m.sum(), 0.0);
    EXPECT_DOUBLE_EQ(m.mean(), 0.0);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_DOUBLE_EQ(m(i, j), 0.0);
}

// Test addition with zero matrix
TEST(MatrixEdgeCaseTest, AdditionWithZeroMatrix)
{
    Matrix m1(2, 2);
    m1(0, 0) = 1.0; m1(0, 1) = 2.0;
    m1(1, 0) = 3.0; m1(1, 1) = 4.0;

    Matrix m2(2, 2);

    Matrix result = m1 + m2;
    EXPECT_DOUBLE_EQ(result(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 4.0);
}

// Test multiplication with identity matrix
TEST(MatrixEdgeCaseTest, MultiplicationWithIdentityMatrix)
{
    Matrix m1(2, 2);
    m1(0, 0) = 3.0; m1(0, 1) = 4.0;
    m1(1, 0) = 5.0; m1(1, 1) = 6.0;

    Matrix identity(2, 2);
    identity(0, 0) = 1.0; identity(0, 1) = 0.0;
    identity(1, 0) = 0.0; identity(1, 1) = 1.0;

    Matrix result = m1 * identity;
    EXPECT_DOUBLE_EQ(result(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 5.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 6.0);
}

// Test multiplication by zero scalar
TEST(MatrixEdgeCaseTest, MultiplicationByZeroScalar)
{
    Matrix m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(1, 0) = 3.0; m(1, 1) = 4.0;

    Matrix result = m * 0.0;
    EXPECT_DOUBLE_EQ(result(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 0.0);
}

// Test multiplication by negative scalar
TEST(MatrixEdgeCaseTest, MultiplicationByNegativeScalar)
{
    Matrix m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(1, 0) = 3.0; m(1, 1) = 4.0;

    Matrix result = m * -2.0;
    EXPECT_DOUBLE_EQ(result(0, 0), -2.0);
    EXPECT_DOUBLE_EQ(result(0, 1), -4.0);
    EXPECT_DOUBLE_EQ(result(1, 0), -6.0);
    EXPECT_DOUBLE_EQ(result(1, 1), -8.0);
}

// Test addition with negative scalar
TEST(MatrixEdgeCaseTest, AdditionWithNegativeScalar)
{
    Matrix m(2, 2);
    m(0, 0) = 5.0; m(0, 1) = 4.0;
    m(1, 0) = 3.0; m(1, 1) = 2.0;

    Matrix result = m + (-3.0);
    EXPECT_DOUBLE_EQ(result(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(result(1, 1), -1.0);
}

// Test with very large values
TEST(MatrixEdgeCaseTest, VeryLargeValues)
{
    Matrix m(2, 2);
    double largeVal = 1e10;
    m(0, 0) = largeVal;
    m(0, 1) = largeVal;
    m(1, 0) = largeVal;
    m(1, 1) = largeVal;

    double sum = m.sum();
    EXPECT_DOUBLE_EQ(sum, 4 * largeVal);
}

// Test with very small values
TEST(MatrixEdgeCaseTest, VerySmallValues)
{
    Matrix m(2, 2);
    double smallVal = 1e-10;
    m(0, 0) = smallVal;
    m(0, 1) = smallVal;
    m(1, 0) = smallVal;
    m(1, 1) = smallVal;

    double sum = m.sum();
    EXPECT_NEAR(sum, 4 * smallVal, 1e-15);
}

// Test with negative values
TEST(MatrixEdgeCaseTest, NegativeValues)
{
    Matrix m(2, 2);
    m(0, 0) = -1.0; m(0, 1) = -2.0;
    m(1, 0) = -3.0; m(1, 1) = -4.0;

    EXPECT_DOUBLE_EQ(m.sum(), -10.0);
    EXPECT_DOUBLE_EQ(m.mean(), -2.5);
}

// Test Hadamard product with negative values
TEST(MatrixEdgeCaseTest, HadamardProductNegativeValues)
{
    Matrix m1(2, 2);
    m1(0, 0) = -2.0; m1(0, 1) = 3.0;
    m1(1, 0) = -4.0; m1(1, 1) = 5.0;

    Matrix m2(2, 2);
    m2(0, 0) = 2.0; m2(0, 1) = -3.0;
    m2(1, 0) = 4.0; m2(1, 1) = -5.0;

    Matrix result = m1.hadamardProduct(m2);
    EXPECT_DOUBLE_EQ(result(0, 0), -4.0);
    EXPECT_DOUBLE_EQ(result(0, 1), -9.0);
    EXPECT_DOUBLE_EQ(result(1, 0), -16.0);
    EXPECT_DOUBLE_EQ(result(1, 1), -25.0);
}

// Test sumRows with single column
TEST(MatrixEdgeCaseTest, SumRowsSingleColumn)
{
    Matrix m(3, 1);
    m(0, 0) = 1.0;
    m(1, 0) = 2.0;
    m(2, 0) = 3.0;

    Matrix result = m.sumRows();
    EXPECT_EQ(result.getRows(), 3);
    EXPECT_EQ(result.getCols(), 1);
    EXPECT_DOUBLE_EQ(result(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(result(2, 0), 3.0);
}

// Test sumRows with single row
TEST(MatrixEdgeCaseTest, SumRowsSingleRow)
{
    Matrix m(1, 5);
    m(0, 0) = 1.0; m(0, 1) = 2.0; m(0, 2) = 3.0;
    m(0, 3) = 4.0; m(0, 4) = 5.0;

    Matrix result = m.sumRows();
    EXPECT_EQ(result.getRows(), 1);
    EXPECT_EQ(result.getCols(), 1);
    EXPECT_DOUBLE_EQ(result(0, 0), 15.0);
}

// Test map with identity function
TEST(MatrixEdgeCaseTest, MapIdentityFunction)
{
    Matrix m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(1, 0) = 3.0; m(1, 1) = 4.0;

    Matrix result = m.map([](double x) { return x; });
    EXPECT_DOUBLE_EQ(result(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 4.0);
}

// Test map with function that returns zero
TEST(MatrixEdgeCaseTest, MapZeroFunction)
{
    Matrix m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(1, 0) = 3.0; m(1, 1) = 4.0;

    Matrix result = m.map([](double x) { return 0.0; });
    EXPECT_DOUBLE_EQ(result(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 0.0);
}

// Test applyFunction with negation function
TEST(MatrixEdgeCaseTest, ApplyFunctionNegation)
{
    Matrix m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = -2.0;
    m(1, 0) = 3.0; m(1, 1) = -4.0;

    m.applyFunction([](double x) { return -x; });
    EXPECT_DOUBLE_EQ(m(0, 0), -1.0);
    EXPECT_DOUBLE_EQ(m(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m(1, 0), -3.0);
    EXPECT_DOUBLE_EQ(m(1, 1), 4.0);
}

// Test assignment operator with self-assignment
TEST(MatrixEdgeCaseTest, SelfAssignment)
{
    Matrix m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(1, 0) = 3.0; m(1, 1) = 4.0;

    m = m;

    EXPECT_EQ(m.getRows(), 2);
    EXPECT_EQ(m.getCols(), 2);
    EXPECT_DOUBLE_EQ(m(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m(1, 1), 4.0);
}

// Test multiplication with minimum dimensions (vectors)
TEST(MatrixEdgeCaseTest, VectorMultiplication)
{
    Matrix rowVec(1, 3);
    rowVec(0, 0) = 1.0; rowVec(0, 1) = 2.0; rowVec(0, 2) = 3.0;

    Matrix colVec(3, 1);
    colVec(0, 0) = 4.0; colVec(1, 0) = 5.0; colVec(2, 0) = 6.0;

    Matrix result = rowVec * colVec;
    EXPECT_EQ(result.getRows(), 1);
    EXPECT_EQ(result.getCols(), 1);
    EXPECT_DOUBLE_EQ(result(0, 0), 32.0);
}

// Test with tall rectangular matrix (10x2)
TEST(MatrixEdgeCaseTest, TallMatrix)
{
    Matrix m(10, 2);
    for (int i = 0; i < 10; i++)
    {
        m(i, 0) = i * 1.0;
        m(i, 1) = i * 2.0;
    }

    Matrix result = m.transpose();
    EXPECT_EQ(result.getRows(), 2);
    EXPECT_EQ(result.getCols(), 10);

    for (int i = 0; i < 10; i++)
    {
        EXPECT_DOUBLE_EQ(result(0, i), i * 1.0);
        EXPECT_DOUBLE_EQ(result(1, i), i * 2.0);
    }
}

// Test with wide rectangular matrix (2x10)
TEST(MatrixEdgeCaseTest, WideMatrix)
{
    Matrix m(2, 10);
    for (int j = 0; j < 10; j++)
    {
        m(0, j) = j * 1.0;
        m(1, j) = j * 2.0;
    }

    Matrix result = m.transpose();
    EXPECT_EQ(result.getRows(), 10);
    EXPECT_EQ(result.getCols(), 2);

    for (int j = 0; j < 10; j++)
    {
        EXPECT_DOUBLE_EQ(result(j, 0), j * 1.0);
        EXPECT_DOUBLE_EQ(result(j, 1), j * 2.0);
    }
}

// Test randomize with identical range (min == max)
TEST(MatrixEdgeCaseTest, RandomizeIdenticalRange)
{
    Matrix m(3, 3);
    m.randomize(5.0, 5.0);

    for (int i = 0; i < m.getRows(); i++)
    {
        for (int j = 0; j < m.getCols(); j++)
        {
            EXPECT_NEAR(m(i, j), 5.0, 0.001);
        }
    }
}

// Test with chained operations
TEST(MatrixEdgeCaseTest, ChainedOperations)
{
    Matrix m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(1, 0) = 3.0; m(1, 1) = 4.0;

    Matrix result = ((m * 2.0) + 3.0);

    EXPECT_DOUBLE_EQ(result(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 7.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 9.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 11.0);
}

// Test double transpose (should return original matrix)
TEST(MatrixEdgeCaseTest, DoubleTranspose)
{
    Matrix m(2, 3);
    m(0, 0) = 1.0; m(0, 1) = 2.0; m(0, 2) = 3.0;
    m(1, 0) = 4.0; m(1, 1) = 5.0; m(1, 2) = 6.0;

    Matrix result = m.transpose().transpose();

    EXPECT_EQ(result.getRows(), 2);
    EXPECT_EQ(result.getCols(), 3);
    EXPECT_DOUBLE_EQ(result(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(result(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(result(1, 2), 6.0);
}
