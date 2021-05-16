import unittest, sys
import numpy as np

sys.path.insert(1,'../src')
import dataload, linear

class LinearRegressionCase(unittest.TestCase):
    def TestSlope(self):
        x_t = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
        y_t = np.array([2.5, 5., 7.5, 10., 12.5, 15., 17.5, 20., 22.5])

        linear_regression_test = linear.LINEAR_RSQUARE(x_t, y_t)
        m_t, b_t = linear_regression_test.slope_intercept()
        self.asserAlmostEqual(m_t, 2.5);

    def TestIntercept(self):
        x_t = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
        y_t = np.array([2.5, 5., 7.5, 10., 12.5, 15., 17.5, 20., 22.5])

        linear_regression_test = linear.LINEAR_RSQUARE(x_t, y_t)
        m_t, b_t = linear_regression_test.slope_intercept()
        self.asserAlmostEqual(b_t, 0.0);

if __name__ == '__main__':

    unittest.main()
    # print("Slope: {} Intercept: {}".format(m_t, b_t) )
    


