#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>

// Update the include path if your header is in a different folder
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
