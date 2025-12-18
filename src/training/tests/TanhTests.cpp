#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "../include/Matrix.h"
#include "../include/activation_functions/Tanh.h"

static constexpr double EPS = 1e-12;
static constexpr double LARGE = 1e6;

// Ensure activate matches the standard library tanh for representative values
TEST(TanhTest, MatchesStdTanhKnownValues) {
    Tanh t;
    std::vector<double> xs = {-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0};
    for (double x : xs) {
        ASSERT_NEAR(t.activate(x), std::tanh(x), 1e-12) << "Mismatch at x=" << x;
    }
}

// Tanh should be an odd function: tanh(-x) == -tanh(x)
TEST(TanhTest, OddFunctionProperty) {
    Tanh t;
    std::vector<double> xs = {0.1, 0.5, 1.0, 2.0, 10.0};
    for (double x : xs) {
        ASSERT_NEAR(t.activate(-x), -t.activate(x), EPS) << "Oddness failed at x=" << x;
    }
}

// For very large magnitude inputs, tanh approaches +/-1
TEST(TanhTest, LimitsToPlusMinusOne) {
    Tanh t;
    double p = t.activate(LARGE);
    double n = t.activate(-LARGE);
    ASSERT_NEAR(p, 1.0, 1e-12);
    ASSERT_NEAR(n, -1.0, 1e-12);
}

// Tanh is strictly increasing: check monotonicity on a grid
TEST(TanhTest, MonotonicIncreasing) {
    Tanh t;
    std::vector<double> xs = {-5.0, -2.0, -1.0, -0.1, 0.0, 0.1, 1.0, 2.0, 5.0};
    for (size_t i = 1; i < xs.size(); ++i) {
        ASSERT_LE(t.activate(xs[i-1]), t.activate(xs[i])) << "Not increasing between " << xs[i-1] << " and " << xs[i];
    }
}

// Verify derivative implementation equals 1 - tanh(x)^2 (analytic formula)
TEST(TanhTest, DerivativeAnalyticMatches) {
    Tanh t;
    std::vector<double> xs = {-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0};
    for (double x : xs) {
        double act = t.activate(x);
        double expected = 1.0 - act * act; // derivative of tanh
        ASSERT_NEAR(t.derivative(x), expected, 1e-12) << "Derivative analytic mismatch at x=" << x;
    }
}

// Cross-check analytic derivative against a central finite difference approximation
TEST(TanhTest, DerivativeMatchesNumerical) {
    Tanh t;
    std::vector<double> xs = {-2.0, -0.5, -1e-3, 0.0, 1e-3, 0.5, 2.0};
    double h = 1e-6;
    for (double x : xs) {
        double fph = t.activate(x + h);
        double fmh = t.activate(x - h);
        double numeric = (fph - fmh) / (2.0 * h);
        double analytic = t.derivative(x);
        // Numeric errors can be larger; allow looser tolerance
        ASSERT_NEAR(analytic, numeric, 1e-6) << "Numerical derivative mismatch at x=" << x;
    }
}

// Test the forward pass with a matrix input
TEST(TanhTest, ForwardPass) {
    Tanh t;
    Matrix m(2, 2);
    m(0, 0) = 0.0;
    m(0, 1) = 1.0;
    m(1, 0) = -1.0;
    m(1, 1) = 2.0;

    Matrix result = t.forward(m);

    Matrix expected(2, 2);
    expected(0, 0) = std::tanh(0.0);
    expected(0, 1) = std::tanh(1.0);
    expected(1, 0) = std::tanh(-1.0);
    expected(1, 1) = std::tanh(2.0);

    for (size_t i = 0; i < m.getRows(); ++i) {
        for (size_t j = 0; j < m.getCols(); ++j) {
            ASSERT_NEAR(result(i, j), expected(i, j), EPS);
        }
    }
}

// Test the backward pass with a matrix input
TEST(TanhTest, BackwardPass) {
    Tanh t;
    Matrix upstreamGradient(2, 2);
    upstreamGradient(0, 0) = 0.5;
    upstreamGradient(0, 1) = 1.5;
    upstreamGradient(1, 0) = 2.5;
    upstreamGradient(1, 1) = 3.5;

    // The backward pass expects the *INPUT* to the activation function
    Matrix activationInput(2, 2);
    activationInput(0, 0) = 0.0;
    activationInput(0, 1) = 1.0;
    activationInput(1, 0) = -2.0;
    activationInput(1, 1) = 3.0;

    Matrix result = t.backward(upstreamGradient, activationInput);

    Matrix expected(2, 2);
    // The derivative of tanh(x) is 1 - tanh(x)^2.
    // The backward pass computes: upstream_grad * derivative(input)
    expected(0, 0) = upstreamGradient(0, 0) * t.derivative(activationInput(0, 0));
    expected(0, 1) = upstreamGradient(0, 1) * t.derivative(activationInput(0, 1));
    expected(1, 0) = upstreamGradient(1, 0) * t.derivative(activationInput(1, 0));
    expected(1, 1) = upstreamGradient(1, 1) * t.derivative(activationInput(1, 1));

    for (size_t i = 0; i < result. getRows(); ++i) {
        for (size_t j = 0; j < result.getCols(); ++j) {
            ASSERT_NEAR(result(i, j), expected(i, j), EPS);
        }
    }
}