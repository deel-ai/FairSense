from fairness_problem import FairnessProblem

def visu_demo_sobol(fairness_problem : FairnessProblem):
    print(fairness_problem.get_result()[["S", "ST", "S_ind", "ST_ind"]])