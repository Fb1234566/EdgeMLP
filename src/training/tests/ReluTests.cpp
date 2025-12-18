#include <gtest/gtest.h>
#include <vector>
#include "../include/Matrix.h"
#include "../include/activation_functions/Relu.h"

static constexpr double EPS = 1e-12;
static constexpr double LARGE = 1e6;

TEST(ReluTest, KnownValues) {
    Relu r;
    // Known behavior: negative -> 0, zero -> 0, positive -> identity
    ASSERT_NEAR(r.activate(-1.0), 0.0, EPS);
    ASSERT_NEAR(r.activate(0.0), 0.0, EPS);
    ASSERT_NEAR(r.activate(1.0), 1.0, EPS);
    ASSERT_NEAR(r.activate(2.5), 2.5, EPS);
}

TEST(ReluTest, Limits) {
    Relu r;
    // For very large positive inputs the output should be approximately the input
    double big = r.activate(LARGE);
    ASSERT_NEAR(big, LARGE, 1e-6);

    // For very large negative inputs the output should be 0
    double neg = r.activate(-LARGE);
    ASSERT_NEAR(neg, 0.0, EPS);
}

TEST(ReluTest, MonotonicityNonDecreasing) {
    Relu r;
    // ReLU is non-decreasing: ensure outputs do not decrease on increasing inputs
    std::vector<double> xs = {-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0};
    for (size_t i = 1; i < xs.size(); ++i) {
        ASSERT_LE(r.activate(xs[i-1]), r.activate(xs[i])) << "Non-decreasing failed between " << xs[i-1] << " and " << xs[i];
    }
}

TEST(ReluTest, DerivativeMatchesExpected) {
    Relu r;
    // We assume derivative = 1 for x > 0 and derivative = 0 for x <= 0
    std::vector<double> xs = {-3.0, -1.0, -1e-6, 0.0, 1e-6, 1.0, 5.0};
    for (double x : xs) {
        double d = r.derivative(x);
        double expected = (x > 0.0) ? 1.0 : 0.0; // assume derivative 0 at x == 0
        ASSERT_NEAR(d, expected, 1e-9) << "Derivative mismatch at x=" << x;
    }
}

TEST(ReluTest, ForwardPass) {
    Relu r;
    Matrix m(2, 2);
    m(0, 0) = 1.0;
    m(0, 1) = -2.0;
    m(1, 0) = 0.0;
    m(1, 1) = 3.5;

    Matrix result = r.forward(m);

    Matrix expected(2, 2);
    expected(0, 0) = 1.0;
    expected(0, 1) = 0.0;
    expected(1, 0) = 0.0;
    expected(1, 1) = 3.5;

    for (size_t i = 0; i < m.getRows(); ++i) {
        for (size_t j = 0; j < m.getCols(); ++j) {
            ASSERT_NEAR(result(i, j), expected(i, j), EPS);
        }
    }
}

TEST(ReluTest, BackwardPass) {
    Relu r;
    Matrix upstreamGradient(2, 2);
    upstreamGradient(0, 0) = 0.5;
    upstreamGradient(0, 1) = 1.5;
    upstreamGradient(1, 0) = 2.5;
    upstreamGradient(1, 1) = 3.5;

    Matrix activationInput(2, 2);
    activationInput(0, 0) = 4.0;   // x > 0, derivative = 1
    activationInput(0, 1) = -1.0;  // x < 0, derivative = 0
    activationInput(1, 0) = 0.0;   // x = 0, derivative = 0 (convention)
    activationInput(1, 1) = 1.0;   // x > 0, derivative = 1

    Matrix result = r.backward(upstreamGradient, activationInput);

    Matrix expected(2, 2);
    expected(0, 0) = 0.5; // derivative is 1, gradient passes through
    expected(0, 1) = 0.0; // derivative is 0, gradient is zeroed out
    expected(1, 0) = 0.0; // derivative is 0, gradient is zeroed out
    expected(1, 1) = 3.5; // derivative is 1, gradient passes through

    for (size_t i = 0; i < result.getRows(); ++i) {
        for (size_t j = 0; j < result.getCols(); ++j) {
            ASSERT_NEAR(result(i, j), expected(i, j), EPS);
        }
    }
}