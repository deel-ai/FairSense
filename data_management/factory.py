from fairness_problem import FairnessProblem

def create_fairness_problem(inputs=None, function=None, outputs=None, labels=None):
    # Etape de vérification du format des entrées ici ?
    return FairnessProblem(inputs, function, outputs, labels)