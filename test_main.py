import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import unittest
import sklearn.datasets
import hw1.linear_regression as hw1linreg

import sklearn.preprocessing
import sklearn.pipeline


plt.rcParams.update({'font.size': 14})
np.random.seed(42)
test = unittest.TestCase()


def evaluate_accuracy(y: np.ndarray, y_pred: np.ndarray):
    """
    Calculates mean squared error (MSE) and coefficient of determination (R-squared).
    :param y: Target values.
    :param y_pred: Predicted values.
    :return: A tuple containing the MSE and R-squared values.
    """
    mse = hw1linreg.mse_score(y, y_pred)
    rsq = hw1linreg.r2_score(y, y_pred)
    return mse, rsq


if __name__ == '__main__':

    # Load data we'll work with - Boston housing dataset
    # We'll use sklearn's built-in data
    ds_boston = sklearn.datasets.load_boston()
    feature_names = ds_boston.feature_names

    n_features = len(feature_names)
    x, y = ds_boston.data, ds_boston.target
    n_samples = len(y)
    print(f'Loaded {n_samples} samples')
    #   Load into a pandas dataframe and show some samples
    df_boston = pd.DataFrame(data=x, columns=ds_boston.feature_names)
    df_boston = df_boston.assign(MEDV=y)
    df_boston.head(10).style.background_gradient(subset=['MEDV'], high=1.)

    n_top_features = 5
    top_feature_names, top_corr = hw1linreg.top_correlated_features(df_boston, 'MEDV', n_top_features)
    print('Top features: ', top_feature_names)
    print('Top features correlations: ', top_corr)

    # Tests
    test.assertEqual(len(top_feature_names), n_top_features)
    test.assertEqual(len(top_corr), n_top_features)
    test.assertAlmostEqual(np.sum(np.abs(top_corr)), 2.893, delta=1e-3)  # compare to precomputed value for n=5

    # # Test BiasTrickTransformer
    # bias_tf = hw1linreg.BiasTrickTransformer()
    #
    # test_cases = [
    #     np.random.randint(10, 20, size=(5, 2)),
    #     np.random.randn(10, 1),
    # ]
    #
    # for xt in test_cases:
    #     xb = bias_tf.fit_transform(xt)
    #     print(xb.shape)
    #
    #     test.assertEqual(xb.ndim, 2)
    #     test.assertTrue(np.all(xb[:, 0] == 1))
    #     test.assertTrue(np.all(xb[:, 1:] == xt))

# Create our model as a pipline:
# First we scale each feature, then the bias trick is applied, then the regressor
model = sklearn.pipeline.make_pipeline(
    sklearn.preprocessing.StandardScaler(),
    hw1linreg.BiasTrickTransformer(),
    hw1linreg.LinearRegressor(),
)
fig, ax = plt.subplots(nrows=1, ncols=n_top_features, sharey=True, figsize=(20, 5))
actual_mse = []

# Fit a single feature at a time
for i, feature_name in enumerate(top_feature_names):
    xf = df_boston[feature_name].values.reshape(-1, 1)

    y_pred = model.fit_predict(xf, y)
    mse, rsq = evaluate_accuracy(y, y_pred)

    x_line = np.arange(xf.min(), xf.max(), 0.1, dtype=np.float).reshape(-1, 1)
    y_line = model.predict(x_line)

    # Plot
    ax[i].scatter(xf, y, marker='o', edgecolor='black')
    ax[i].plot(x_line, y_line, color='red', lw=2, label=f'fit, $R^2={rsq:.2f}$')
    ax[i].set_ylabel('MEDV')
    ax[i].set_xlabel(feature_name)
    ax[i].legend()

    actual_mse.append(mse)

# Test regressor implementation
print(actual_mse)
expected_mse = [38.862, 43.937, 62.832, 64.829, 66.040]
for i in range(len(expected_mse)):
    test.assertAlmostEqual(expected_mse[i], actual_mse[i], delta=1e-1)