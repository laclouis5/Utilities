import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression

def fit_plane(X, Y, Z, mask=None):
    """
    Fits a plane to observed data corrupted with noise:
        Z = a*X + b*Y + c + e

    Paramters:
        X, Y (array<float> of ndim=2):
            Arrays containing the coordinates of the point
            in the 2-D space.
        Z (array<float> of ndim=2):
            Observed values of the function.
        mask (optional array<bool> of ndim=2):
            Mask where to keep data.

    Returns:
        (array of size 3):
            The coefficients of the fitted plane.
            Z = a*X + b*Y + c + e  ->  [a, b, c]
    """
    if mask is not None:
        X = X[mask == True]
        Y = Y[mask == True]
        Z = Z[mask == True]
        T = np.stack((X, Y), axis=1)
        V = Z
    else:
        T = np.stack((X, Y), axis=2).reshape(-1, 2)
        V = Z.reshape(-1)

    reg = LinearRegression().fit(T, V)
    coeffs = [*reg.coef_, reg.intercept_]

    return coeffs

def reg_score(y, f, mask=None):
    """
    Parameters:
        y(N-D array): the observed data
        f(N-D array): the predicted data

    Returns:
        (float): the R2 value
    """
    if mask is not None:
        y = y[mask == True]
        f = f[mask == True]

    y_mean = np.mean(y)
    SStot = np.sum(np.square(y - y_mean))
    SSres = np.sum(np.square(y - f))

    return 1 - SSres / SStot

if __name__ == "__main__":
    width, height = (100, 50)
    a, b, c = 2, 3, 7

    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    noise = 10 * np.random.randn(height, width)
    Z = a*X + b*Y + c + noise

    mask = np.full_like(Z, False)  # Simulate missing data
    mask[::4, ::4] = True

    # Fit the data
    coeffs = fit_plane(X, Y, Z, mask)

    # Predicted Plane
    H = coeffs[0] * X + coeffs[1] * Y + coeffs[2]

    print("Predicted Plane: Z = {:.4}*X + {:.4}*Y + {:.4}".format(*coeffs))
    print("Real Plane: Z = {}*X + {}*Y + {}".format(a, b, c))
    print("R²: {:.2%}".format(reg_score(Z, H, mask)))

    # Plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(X, Y, Z, s=1)
    ax.plot_surface(X, Y, H, color="orange")
    plt.show()
