import numpy as np
import pandas as pd
from deel.fairsense.indices import sobol_indices, cvm_indices
from deel.fairsense.utils.dataclasses import IndicesInput
from deel.fairsense.utils.fairness_objective import y_pred


def gaussian_data_generator(sigma12, sigma13, sigma23, N, var1=1.0, var2=1.0, var3=1.0):
    cov = np.mat(
        [[var1, sigma12, sigma13], [sigma12, var2, sigma23], [sigma13, sigma23, var3]]
    )
    x = np.random.multivariate_normal(mean=np.array([0, 0, 0]), cov=cov, size=N)
    return pd.DataFrame(x, columns=[0, 1, 2])


def run_experiment(nsample, function, data_generator, data_generator_kwargs):
    data = data_generator(**data_generator_kwargs)
    inputs = IndicesInput(model=function[0], x=data, objective=y_pred)
    results = sobol_indices(inputs, n=nsample) + cvm_indices(inputs)
    return results


def test_output_scaling():
    nsample = 1 * 10 ** 5
    data_sample = 1 * 10 ** 4
    func1 = lambda x: x["0"] + 0.01 * x["1"] + 0.01 * x["2"], "f(x) -> X_0"
    func1bis = lambda x: 10 * x["0"] + 0.01 * x["1"] + 0.01 * x["2"], "f(x) -> 10*X_0"
    results = run_experiment(
        function=func1,
        nsample=nsample,
        data_generator=gaussian_data_generator,
        data_generator_kwargs=dict(
            sigma12=0.0, sigma13=0.0, sigma23=0.0, N=data_sample
        ),
    )
    target = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    print(results.values, target)
    np.testing.assert_allclose(results.values, target, atol=1e-1)
    results = run_experiment(
        function=func1bis,
        nsample=nsample,
        data_generator=gaussian_data_generator,
        data_generator_kwargs=dict(
            sigma12=0.0, sigma13=0.0, sigma23=0.0, N=data_sample
        ),
    )
    target = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    print(results.values, target)
    np.testing.assert_allclose(results.values, target, atol=1e-1)


def test_input_scaling():
    nsample = 1 * 10 ** 5
    data_sample = 1 * 10 ** 4
    func1 = lambda x: x["0"] + 0.01 * x["1"] + 0.01 * x["2"], "f(x) -> X_0"
    results = run_experiment(
        function=func1,
        nsample=nsample,
        data_generator=gaussian_data_generator,
        data_generator_kwargs=dict(
            sigma12=0.0, sigma13=0.0, sigma23=0.0, var1=10.0, N=data_sample
        ),
    )
    target = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    print(results.values, target)
    np.testing.assert_allclose(results.values, target, atol=1e-1)


def test_correlations():
    nsample = 1 * 10 ** 5
    data_sample = 1 * 10 ** 4
    func1 = lambda x: x["0"] + 0.001 * x["1"] + 0.001 * x["2"], "f(x) -> X_0"
    results = run_experiment(
        function=func1,
        nsample=nsample,
        data_generator=gaussian_data_generator,
        data_generator_kwargs=dict(
            sigma12=0.6, sigma13=0.6, sigma23=0.0, N=data_sample
        ),
    )
    target = [
        [1.0, 1.0, 0.27, 0.27, 0.5, 1.0],
        [0.35, 0.35, 0.0, 0.0, 0.0, 0.25],
        [0.35, 0.35, 0.0, 0.0, 0.0, 0.25],
    ]
    print(results.values, target)
    np.testing.assert_allclose(results.values, target, rtol=1e-1, atol=1e-1)


def test_distribution():
    nsample = 1 * 10 ** 5
    data_sample = 1 * 10 ** 4
    func3 = lambda x: x["0"] + x["1"] + 0.01 * x["2"], "f(x) -> X_0 + X_1"
    results = run_experiment(
        function=func3,
        nsample=nsample,
        data_generator=gaussian_data_generator,
        data_generator_kwargs=dict(
            sigma12=0.0, sigma13=0.0, sigma23=0.0, var1=1.0, N=data_sample
        ),
    )
    target = [
        [0.5, 0.5, 0.5, 0.5, 0.66, 0.33],
        [0.5, 0.5, 0.5, 0.5, 0.66, 0.33],
        [0.0, 0.0, -0.0, 0.0, 0.0, 0.0],
    ]
    print(results.values, target)
    np.testing.assert_allclose(results.values, target, rtol=1e-1, atol=1e-1)
    results = run_experiment(
        function=func3,
        nsample=nsample,
        data_generator=gaussian_data_generator,
        data_generator_kwargs=dict(
            sigma12=0.0, sigma13=0.0, sigma23=0.0, var1=10.0, N=data_sample
        ),
    )
    target = [
        [0.85, 0.85, 0.85, 0.85, 0.9, 0.72],
        [0.15, 0.15, 0.15, 0.15, 0.3, 0.],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    print(results.values, target)
    np.testing.assert_allclose(results.values, target, rtol=1e-1, atol=1e-1)


def test_joint_effects():
    nsample = 1 * 10 ** 5
    data_sample = 1 * 10 ** 4
    func4 = (
        lambda x: x["0"] * (((x["1"] > 0) * (x["2"] > 0) * 20) + -10),
        "f(x) -> 20*X_0 if (X_1 > 0.5) && (X_2 > 0.5) else: 0.25*X_0 ",
    )
    results = run_experiment(
        function=func4,
        nsample=nsample,
        data_generator=gaussian_data_generator,
        data_generator_kwargs=dict(
            sigma12=0.0, sigma13=0.0, sigma23=0.0, var1=1.0, N=data_sample
        ),
    )
    target = [
        [0.25, 1.0, 0.25, 1.0, 1.0, 0.5],
        [0.0, 0.5, 0.0, 0.50, 0.3, 0.0],
        [0.0, 0.5, 0.0, 0.50, 0.3, 0.0],
    ]
    print(results.values, target)
    np.testing.assert_allclose(results.values, target, rtol=1e-1, atol=1e-1)
