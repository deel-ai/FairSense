
class FairnessProblem():

    def __init__(self, inputs=None, function=None, outputs=None, labels=None):
        self.inputs = inputs
        self.function = function
        self.outputs = outputs
        self.labels = labels
        self.result = None
    
    def get_result(self):
        return self.result