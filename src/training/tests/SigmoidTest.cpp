#include <gtest/gtest.h>
#include <cmath>
#include <vector>

// Update the include path if your header is in a different folder
#include "../include/Matrix.h"
#include "../include/activation_functions/Sigmoid.h"

static constexpr double EPS = 1e-7;
static constexpr double LARGE = 40.0;

// Test: known values
TEST(SigmoidTest, KnownValues) {
    Sigmoid s;
    double v0 = s.activate(0.0);
    ASSERT_NEAR(v0, 0.5, 1e-12);

    double v1 = s.activate(1.0);
    double expected1 = 1.0 / (1.0 + std::exp(-1.0));
    ASSERT_NEAR(v1, expected1, 1e-12);

    double vn1 = s.activate(-1.0);
    double expected_n1 = 1.0 / (1.0 + std::exp(1.0));
    ASSERT_NEAR(vn1, expected_n1, 1e-12);
}

// Test: limits for very large / very negative inputs
TEST(SigmoidTest, Limits) {
    Sigmoid s;
    double sbig = s.activate(LARGE);
    ASSERT_GT(sbig, 1.0 - 1e-12);
    ASSERT_LT(sbig, 1.0 + 1e-12);

    double sneg = s.activate(-LARGE);
    ASSERT_GT(sneg, 0.0 - 1e-12);
    ASSERT_LT(sneg, 1e-12);
}

// Test: monotonicity on several points
TEST(SigmoidTest, Monotonicity) {
    Sigmoid s;
    std::vector<double> xs = {-3.0, -1.0, 0.0, 1.0, 3.0};
    for (size_t i = 1; i < xs.size(); ++i) {
        ASSERT_LT(s.activate(xs[i-1]), s.activate(xs[i])) << "Monotonicity failed between " << xs[i-1] << " and " << xs[i];
    }
}

// Test: analytic derivative compared with s*(1-s)
TEST(SigmoidTest, DerivativeMatchesFormula) {
    Sigmoid s;
    for (double x = -6.0; x <= 6.0; x += 0.5) {
        double out = s.activate(x);
        double expected = out * (1.0 - out);
        double d = s.derivative(x);
        ASSERT_NEAR(d, expected, 1e-6) << "Derivative mismatch at x=" << x;
    }
}

// Test: forward pass with a matrix
TEST(SigmoidTest, ForwardPass) {
    Sigmoid s;
    Matrix m(2, 2);
    m(0, 0) = 0.0;
    m(0, 1) = 1.0;
    m(1, 0) = -1.0;
    m(1, 1) = 2.0;

    Matrix result = s.forward(m);

    Matrix expected(2, 2);
    expected(0, 0) = s.activate(0.0);
    expected(0, 1) = s.activate(1.0);
    expected(1, 0) = s.activate(-1.0);
    expected(1, 1) = s.activate(2.0);

    for (size_t i = 0; i < m.getRows(); ++i) {
        for (size_t j = 0; j < m.getCols(); ++j) {
            ASSERT_NEAR(result(i, j), expected(i, j), EPS);
        }
    }
}

// Test: backward pass with a matrix
TEST(SigmoidTest, BackwardPass) {
    Sigmoid s;
    Matrix upstreamGradient(2, 2);
    upstreamGradient(0, 0) = 0.5;
    upstreamGradient(0, 1) = 1.5;
    upstreamGradient(1, 0) = 2.5;
    upstreamGradient(1, 1) = 3.5;

    // Pass INPUT values, not outputs
    Matrix activationInput(2, 2);
    activationInput(0, 0) = 0.0;
    activationInput(0, 1) = 1.0;
    activationInput(1, 0) = -2.0;
    activationInput(1, 1) = 3.0;

    Matrix result = s.backward(upstreamGradient, activationInput);

    Matrix expected(2, 2);
    expected(0, 0) = upstreamGradient(0, 0) * s.derivative(activationInput(0, 0));
    expected(0, 1) = upstreamGradient(0, 1) * s.derivative(activationInput(0, 1));
    expected(1, 0) = upstreamGradient(1, 0) * s.derivative(activationInput(1, 0));
    expected(1, 1) = upstreamGradient(1, 1) * s.derivative(activationInput(1, 1));

    for (size_t i = 0; i < result.getRows(); ++i) {
        for (size_t j = 0; j < result.getCols(); ++j) {
            ASSERT_NEAR(result(i, j), expected(i, j), EPS);
        }
    }
}