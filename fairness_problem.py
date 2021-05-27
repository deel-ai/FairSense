
class FairnessProblem():

    def __init__(self, inputs=None, function=None, outputs=None, labels=None):
        self.inputs = inputs
        self.function = function
        self.outputs = outputs
        self.labels = labels
        self.result = None
    
    def get_inputs(self):
        return self.inputs

    def get_function(self):
        return self.function

    def get_outputs(self):
        return self.outputs

    def get_labels(self):
        return self.labels

    def get_result(self):
        return self.result
    
    def set_result(self, result):
        self.result = result