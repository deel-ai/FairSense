import indices.sobol
import visualization.display_sobol
import indices.cvm
import visualization.display_cvm

class FairnessProblem():

    def __init__(self, inputs=None, function=None, outputs=None, labels=None):
        self.inputs = inputs
        self.function = function
        self.outputs = outputs
        self.labels = labels

    # Sobol
    def compute_sobol(self, **kargs):
        result = indices.sobol.compute_sobol(self.function, self.inputs, kargs)
        return result

    def display_sobol(self, sobol_result):
        visualization.display_sobol.display_sobol(sobol_result)
        return

    def sobol(self, **kargs):
        res = self.compute_sobol()
        self.display_sobol(res)

    # CVM
    def compute_cvm(self, **kargs):
        result = indices.cvm.compute_cvm(self.inputs, self.outputs)
        return result

    def display_cvm(self, cvm_result):
        visualization.display_cvm.display_cvm(cvm_result)
        return

    def cvm(self, **kargs):
        res = self.compute_cvm()
        self.display_cvm(res)