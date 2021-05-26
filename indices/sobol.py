from fairness_problem import FairnessProblem

def check_arg_sobol(**kargs):
    if False:
        raise Exception()

def compute_sobol(fairness_problem : FairnessProblem, n=1000, N=None, bs=150):
    check_arg_sobol(n=n, N=N, bs=bs)
    # ... calcul ...
    fairness_problem.result = 0