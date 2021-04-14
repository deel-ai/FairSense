import numpy as np


def sobol_unnormalized(zc, zcbis, zcter, f, mode):
    if mode == "on_error":
        rf = lambda x: np.abs(f(x[:, 1:]) - x[:, 0])
    else:
        rf = f
    return np.mean(np.multiply(rf(zc), (rf(zcter) - rf(zcbis))))


def sobol_total_ind_unnormalized(zcbis, zcquad, f, mode):
    if mode == "on_error":
        rf = lambda x: np.abs(f(x[:, 1:]) - x[:, 0])
    else:
        rf = f
    return np.mean(np.square(rf(zcquad) - rf(zcbis)))


def sobol_ind_unnormalized(zc, zcbis, zcquad, f, mode):
    if mode == "on_error":
        rf = lambda x: np.abs(f(x[:, 1:]) - x[:, 0])
    else:
        rf = f
    return np.mean(np.multiply(rf(zc), (rf(zcquad) - rf(zcbis))))


def sobol_total_unnormalized(zcbis, zcter, f, mode):
    if mode == "on_error":
        rf = lambda x: np.abs(f(x[:, 1:]) - x[:, 0])
    else:
        rf = f
    return np.mean(np.square(rf(zcter) - rf(zcbis)))
