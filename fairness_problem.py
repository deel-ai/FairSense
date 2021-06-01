import data_management.factory
import types

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
    
    def set_inputs(self, inputs):
        data_management.factory.check_inputs_type(inputs)
        self.inputs = inputs

    def set_function(self, function):
        data_management.factory.check_function_type(function)
        self.function = function

    def set_outputs(self, outputs):
        data_management.factory.check_outputs_type(outputs)
        self.outputs = outputs

    def set_labels(self, labels):
        data_management.factory.check_labels_type(labels)
        self.labels = labels

    def set_result(self, result):
        # TODO une vérif si formalisation de la sortie
        self.result = result