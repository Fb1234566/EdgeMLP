#include <gtest/gtest.h>
#include "../include/Matrix.h"
#include "../include/loss_functions/MSE.h"

// Test the forward pass of the MSE cost function
TEST(MSETest, ForwardPass)
{
    MSE mse;
    Matrix output(2, 2);
    output(0, 0) = 1.0; output(0, 1) = 2.0;
    output(1, 0) = 3.0; output(1, 1) = 4.0;

    Matrix target(2, 2);
    target(0, 0) = 1.0; target(0, 1) = 1.0;
    target(1, 0) = 1.0; target(1, 1) = 1.0;

    // Squared errors: [0, 1, 4, 9]. Sum = 14. MSE = 14 / 4 = 3.5
    double loss = mse.calculate(output, target);
    EXPECT_DOUBLE_EQ(loss, 3.5);
}

// Test the derivative of the MSE cost function
TEST(MSETest, Derivative)
{
    MSE mse;
    Matrix output(2, 2);
    output(0, 0) = 1.0; output(0, 1) = 2.0;
    output(1, 0) = 3.0; output(1, 1) = 4.0;

    Matrix target(2, 2);
    target(0, 0) = 1.0; target(0, 1) = 1.0;
    target(1, 0) = 1.0; target(1, 1) = 1.0;

    Matrix derivative = mse.derivative(output, target);

    // n = 4, scalar = 2/n = 0.5
    // error = [[0, 1], [2, 3]]
    // derivative = error * 0.5 = [[0, 0.5], [1.0, 1.5]]
    EXPECT_EQ(derivative.getRows(), 2);
    EXPECT_EQ(derivative.getCols(), 2);
    EXPECT_DOUBLE_EQ(derivative(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(derivative(0, 1), 0.5);
    EXPECT_DOUBLE_EQ(derivative(1, 0), 1.0);
    EXPECT_DOUBLE_EQ(derivative(1, 1), 1.5);
}

// Test handling of incompatible dimensions
TEST(MSETest, IncompatibleDimensions)
{
    MSE mse;
    Matrix output(2, 2);
    Matrix target(2, 3); // Different dimensions

    EXPECT_THROW(mse.calculate(output, target), std::invalid_argument);
    EXPECT_THROW(mse.derivative(output, target), std::invalid_argument);
}

// Test with a single element matrix
TEST(MSETest, SingleElement)
{
    MSE mse;
    Matrix output(1, 1);
    output(0, 0) = 5.0;

    Matrix target(1, 1);
    target(0, 0) = 3.0;

    // MSE = (5-3)^2 / 1 = 4
    double loss = mse.calculate(output, target);
    EXPECT_DOUBLE_EQ(loss, 4.0);

    // Derivative = (2/1) * (5-3) = 4
    Matrix derivative = mse.derivative(output, target);
    EXPECT_DOUBLE_EQ(derivative(0, 0), 4.0);
}

// Test case where output equals target
TEST(MSETest, ZeroError)
{
    MSE mse;
    Matrix m(2, 2);
    m(0, 0) = 1.0; m(0, 1) = 2.0;
    m(1, 0) = 3.0; m(1, 1) = 4.0;

    // Loss should be 0
    double loss = mse.calculate(m, m);
    EXPECT_DOUBLE_EQ(loss, 0.0);

    // Derivative should be a zero matrix
    Matrix derivative = mse.derivative(m, m);
    EXPECT_DOUBLE_EQ(derivative.sum(), 0.0);
}
