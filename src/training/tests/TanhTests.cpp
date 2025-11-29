#include <gtest/gtest.h>
#include <vector>
#include <cmath>
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
