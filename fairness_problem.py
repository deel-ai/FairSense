import data_management.factory
import types


class FairnessProblem():

    def __init__(self, inputs=None, columns=None, function=None, outputs=None, labels=None, groups_studied=[], categorical_features=[]):
        self.columns = columns
        self.inputs = inputs
        self.function = function
        self.outputs = outputs
        self.labels = labels
        self.groups_studied = groups_studied
        self.categorical_features = categorical_features
        self.result = None

    def get_inputs(self):
        return self.inputs

    def get_columns(self):
        return self.columns

    def get_function(self):
        return self.function

    def get_outputs(self):
        return self.outputs

    def get_labels(self):
        return self.labels

    def get_groups_studied(self):
        return self.groups_studied

    def get_categorical_features(self):
        return self.categorical_features

    def get_result(self):
        return self.result

    def set_inputs(self, inputs):
        data_management.factory.check_inputs_type(inputs)
        self.inputs = inputs

    def set_columns(self, columns):
        data_management.factory.check_columns_type(columns)
        self.columns = columns

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
    
    def set_categorical_features(self, categorical_features):
        data_management.factory.check_categorical_features_type(categorical_features)
        self.categorical_features = categorical_features

    def set_result(self, result):
        self.result = result
