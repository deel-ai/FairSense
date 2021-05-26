
def check_arg_sobol(**kargs):
    if False:
        raise Exception()

def compute_sobol(f, x, y=None, n=1000, N=None, bs=150):
    check_arg_sobol(f=f, x=x, y=y, n=n, N=N, bs=bs)
    res = 0
    # ... calcul ...
    return res