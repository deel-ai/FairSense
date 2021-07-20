from .data_management.checks import *
import types


class FairnessProblem():

    def __init__(self, inputs=None, columns=None, function=None, outputs=None, labels=None, groups_studied=[], categorical_features=[]):
        self.columns = columns
        self.inputs = inputs
        self.function = function
        self.outputs = outputs
        self.labels = labels
        self.groups_studied = groups_studied
        self.categorical_features = self.set_categorical_features(categorical_features)
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
        check_inputs_type(inputs)
        self.inputs = inputs

    def set_columns(self, columns):
        check_columns_type(columns)
        self.columns = columns

    def set_function(self, function):
        check_function_type(function)
        self.function = function

    def set_outputs(self, outputs):
        check_outputs_type(outputs)
        self.outputs = outputs

    def set_labels(self, labels):
        check_labels_type(labels)
        self.labels = labels

    def set_groups_studied(self, groups_studied):
        check_groups_studied_type(groups_studied)
        self.groups_studied = groups_studied
    
    def set_categorical_features(self, categorical_features):
        check_categorical_features_type(categorical_features)
        temp = categorical_features
        if len(categorical_features) > 0:
            if type(categorical_features[0]) == str:
                temp = []
                col = self.get_columns()
                for elt in categorical_features:
                    try:
                        if col.index(elt) not in temp:
                            temp.append(col.index(elt))
                    except ValueError:
                        raise ValueError("Verify that names you gave to categorical_features are correct : " + str(elt))
        self.categorical_features = temp

    def set_result(self, result):
        self.result = result
