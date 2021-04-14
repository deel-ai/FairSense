from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

samples = 10000
cols = ["X_{}".format(i) for i in range(5)]
x = pd.DataFrame(np.random.normal(size=(samples, 5)), columns=cols)
x["Y"] = x["X_0"] + 2 * x["X_1"] - x["X_2"] + np.random.normal(0, 0.005, len(x))
x = x.sort_values("Y")
x["i"] = range(1, len(x) + 1)


def CVM(data: pd.DataFrame, x_name, z_name, y_name):
    """
    Compute the CVM index T(Y, Z|X)
    Args:
        data: the input data, already sorted by y_name, and with a column containing the Ri named "i"
        x_name: name of the columns in the set of variables X
        z_name: name of the columns in the set of variable Z
        y_name: name of the columns with the variable Y (not used, as we already have a column containing the Ri)

    Returns: the tuple(T(Y, Z), T(Y, Z|X))

    """
    # reformat the column names to have a list
    if isinstance(x_name, str):
        x_name = [x_name]
    elif isinstance(x_name, set):
        x_name = list(x_name)
    if isinstance(z_name, str):
        z_name = [z_name]
    elif isinstance(z_name, set):
        z_name = list(z_name)
    # compute Ni
    # build a KDTree containing the sampling of X
    kd_xi = KDTree(data[x_name])
    # query the tree to get the two closest neighbor of each sample. The closest will be the sample itself,
    # and the second will be the closest neighbors. This returns both the distance with the neighbor and
    # the index of the neighbor in the list used to build the tree.
    dist, ind = kd_xi.query(data[x_name], k=2)
    # compute N_i ( add 1 as python indices start from 0, and the formula use indices starting from 1
    data["N_i"] = ind[:, 1] + 1
    # compute M_i
    # we repeat the same process, but we work on (X,Y) this time
    kd_xiyi = KDTree(data[x_name + z_name])  # build KDTree
    dist, ind = kd_xiyi.query(data[x_name + z_name], k=2)  # find closest neighbors
    data["M_i"] = ind[:, 1] + 1  # save M_i
    # compute L_i
    data["L_i"] = len(data) + 1 - data["i"]  # the + 1 account the fact that Ri indices start from 1
    # compute the M_i used in the equation T(Y,Z)
    kd_zi = KDTree(data[z_name])
    dist, ind = kd_zi.query(data[z_name], k=2)
    data["M_i2"] = ind[:, 1] + 1
    # compute CVM
    n = len(data)
    num_1 = np.sum(np.minimum(data["i"], data["M_i"]) - np.minimum(data["i"], data["N_i"])) / n**2
    den_1 = (np.sum(data["i"] - np.minimum(data["i"], data["N_i"]))) / n**2
    num_2 = np.sum((len(data) * np.minimum(data["i"], data["M_i2"])) - np.square(data["L_i"])) / n**3
    den_2 = np.sum(data["L_i"] * (len(data) - data["L_i"])) / n**3
    tn_ind = num_1 / den_1
    tn_cond = num_2 / den_2
    tn_ind = np.clip(tn_ind, 0., 1.)
    tn_cond = np.clip(tn_cond, 0., 1.)
    u = np.clip(num_1 / den_2, 0., 1.)
    u2 = np.clip(num_2 / den_1, 0., 1.)
    return tn_cond, u


def analyze(x, output_var, cols=None):
    """
    return the CVM indices
    Args:
        x: a dataframe containing inputs and outputs
        output_var: name of the column cotaining the output
        cols: if None one index is computed for each column,
              a list of list containing the columns names can also be provided
              then one index will be computed for each group.
              [["A"], ["B=1", "B=2"]] wil return 2 indices: one for A, and one
              for the group B.

    Returns: a Dataframe with the indices for each group.

    """
    col_was_none = cols is None
    # manipulate a copy of the dataset
    x = x.copy()
    if cols is None:
        cols = list(x.columns).copy()
        cols.remove(output_var)
    # sort the row by increasing Y
    x = x.sort_values(output_var)
    # add a columns with the rank of each row
    x["i"] = range(1, len(x) + 1)
    ## commented code handle ties on the output properly ( same Y => same i )
    ## rows are sorted on Y
    ## compute the difference between two consecutive rows
    # x['i'] = x[output_var].rolling(2).apply(
    #     lambda x: x.index[-1] if (x.iloc[-1] != x.iloc[0]) else np.nan, # if the difference is not null put the value of the second row, else put nan
    # )
    # x['i'].iloc[-1] = len(x['i']) # last row have index len(x)
    # x['i'] = x['i'].bfill() # fill nans with the next filled x

    indices = []
    for col in cols:
        # we compute indices for each variable
        x_names = cols.copy()
        x_names.remove(col)
        try :
            x_names = np.concatenate(x_names).tolist()
        except:
            pass
        indices.append(list(CVM(x.copy(), x_names, col, output_var)) )# + list(CVM(x.copy(), col, x_names, output_var)))
    if col_was_none:
        index = cols
    else:
        ## iterate the list take the first element and keep everything before the "="
        # this turn [["A"], ["B=1","B=2"]] into ["A", "B"]
        index = [x[0].split("=")[0] for x in cols]
    # store the outputs in a dataframe
    df = pd.DataFrame(indices, index=index, columns=["CVM", "CVM_indep"])
    return df


if __name__ == '__main__':
    # quick testing code
    indices = []
    for col in cols:
        # we compute indices for each variable
        x_names = cols.copy()
        x_names.remove(col)
        indices.append(list(CVM(x, x_names, col, "Y")))
    df = pd.DataFrame(indices, index=cols, columns=["CVM", "CVM_indep"])
    print(df.to_markdown(tablefmt="presto", floatfmt=".2f"))

    print("example 2:")
    x = pd.DataFrame(np.random.normal(size=(samples, 5)), columns=cols)
    x["X_0"] = x["X_1"] + np.random.uniform(-0.5, 0.5, len(x))
    x["Y"] = x["X_0"]

    df = analyze(x, "Y")
    print(df.to_markdown(tablefmt="presto", floatfmt=".2f"))
