from sobol_indices.dataset_analyser import analyze
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SobolIndices:
    
    @staticmethod
    def compute(dataset, model, features_names):
        # Create a dataframe with columns names as the dataset features names
        named_dataframe = pd.DataFrame(dataset, columns=features_names)
        # Compute the sobol indices on the dataset with the model 'predict' method
        sobol_indices = analyze(model.predict, named_dataframe, n=1000, bs=75)
        # All indices contain Sobol indices with uncertainty lower/upper bounds
        # Main indices only the 4 main indices (Sobol, Sobol Total, Sobol independent and Sobol Total independent)
        all_indices, main_indices = sobol_indices, sobol_indices[['S', 'ST', 'S_ind', 'ST_ind']]
        return all_indices, main_indices
    
    @staticmethod
    def visualize_barcharts(sobol_indices, columns_names):
        fig, axs = plt.subplots(2, 2, figsize=(16, 8))
        plt.subplots_adjust(wspace=0.1, hspace=0.3)
        plt.suptitle('-- Sobol indices analysis --')
        # Retrieve main Sobol indices
        sobol_values = sobol_indices[['S']].values.squeeze()
        sobol_total_values = sobol_indices[['ST']].values.squeeze()
        sobol_ind_values = sobol_indices[['S_ind']].values.squeeze()
        sobol_total_ind_values = sobol_indices[['ST_ind']].values.squeeze()
        # Display a barchart for Sobol indices
        axs[0, 0].set_title('Sobol')
        axs[0, 0].bar(np.arange(len(sobol_values)), sobol_values)
        axs[0, 0].set_xticks(np.arange(len(columns_names)))
        axs[0, 0].set_xticklabels(columns_names, rotation=90)
        axs[0, 0].plot([0, len(columns_names)-1], [1.0, 1.0], '--r', linewidth=0.5)
        # Display a barchart for Sobol Total indices
        axs[0, 1].set_title('Sobol Total')
        axs[0, 1].bar(np.arange(len(sobol_total_values)), sobol_total_values)
        axs[0, 1].set_xticks(np.arange(len(columns_names)))
        axs[0, 1].set_xticklabels(columns_names, rotation=90)
        axs[0, 1].plot([0, len(columns_names)-1], [1.0, 1.0], '--r', linewidth=0.5)
        # Display a barchart for Sobol independent indices
        axs[1, 0].set_title('Sobol independent')
        axs[1, 0].bar(np.arange(len(sobol_ind_values)), sobol_ind_values)
        axs[1, 0].set_xticks(np.arange(len(columns_names)))
        axs[1, 0].set_xticklabels(columns_names, rotation=90)
        axs[1, 0].plot([0, len(columns_names)-1], [1.0, 1.0], '--r', linewidth=0.5)
        # Display a barchart for Sobol Total independent indices
        axs[1, 1].set_title('Sobol Total independent')
        axs[1, 1].bar(np.arange(len(sobol_total_ind_values)), sobol_total_ind_values)
        axs[1, 1].set_xticks(np.arange(len(columns_names)))
        axs[1, 1].set_xticklabels(columns_names, rotation=90)
        axs[1, 1].plot([0, len(columns_names)-1], [1.0, 1.0], '--r', linewidth=0.5)
        plt.show()
    
    @staticmethod
    def visualize_piecharts(sobol_indices, columns_names):
        # Define an autolabeling function for the piechart
        def label_pie(values):
            def auto_pct(pct):
                val = pct * np.sum(values) / 100.
                return "({:.2f})".format(val)
            return auto_pct
        sobol_values = sobol_indices[['S']].values.squeeze()
        sobol_total_values = sobol_indices[['ST']].values.squeeze()
        sobol_ind_values = sobol_indices[['S_ind']].values.squeeze()
        sobol_total_ind_values = sobol_indices[['ST_ind']].values.squeeze()
        fig, axs = plt.subplots(2, 2, figsize=(16, 16))
        axs[0, 0].set_title('Sobol')
        axs[0, 0].pie(sobol_values, labels=columns_names, autopct=label_pie(sobol_values), shadow=False, normalize=True)
        axs[0, 1].set_title('Sobol Total')
        axs[0, 1].pie(sobol_total_values, labels=columns_names, autopct=label_pie(sobol_total_values), shadow=False, normalize=True)
        axs[1, 0].set_title('Sobol independent')
        axs[1, 0].pie(sobol_ind_values, labels=columns_names, autopct=label_pie(sobol_ind_values), shadow=False, normalize=True)
        axs[1, 1].set_title('Sobol Total independent')
        axs[1, 1].pie(sobol_total_ind_values, labels=columns_names, autopct=label_pie(sobol_total_ind_values), shadow=False, normalize=True)
        plt.show()