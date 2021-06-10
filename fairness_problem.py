import data_management.factory
import types

class FairnessProblem():

    def __init__(self, inputs=None, function=None, outputs=None, labels=None, groups_studied=[]):
        self.inputs = inputs
        self.function = function
        self.outputs = outputs
        self.labels = labels
        self.groups_studied = groups_studied
        self.result = None
    
    def get_inputs(self):
        return self.inputs

    def get_function(self):
        return self.function

    def get_outputs(self):
        return self.outputs

    def get_labels(self):
        return self.labels
    
    def get_groups_studied(self):
        return self.groups_studied

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

    def set_groups_studied(self, groups_studied):
        data_management.factory.check_groups_studied_type(groups_studied)
        self.groups_studied = groups_studied

    def set_result(self, result):
        self.result = result