from fairness_problem import FairnessProblem
import numpy as np

# My data
x = np.linspace(1, 10, 10)
f = lambda x: np.random.randint(0,2)
labels = [0, 1, 1, 0, 1, 0, 0, 0, 1, 1]

# Use of our lib
my_problem = FairnessProblem(inputs=x, function=f, labels=labels)

# Case 1 if we want to itemize
sobol = my_problem.compute_sobol()
my_problem.display_sobol(sobol)

cvm = my_problem.compute_cvm()
my_problem.display_cvm(cvm)

# Otherwise Case 2 
my_problem.sobol()

my_problem.cvm()


