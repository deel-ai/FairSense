from libfairness.fairness_problem import FairnessProblem

def visu_demo_cvm(fairness_problem : FairnessProblem):
    print(fairness_problem.get_result().to_markdown(tablefmt="presto", floatfmt=".2f"))
