from sobol_indices.dataset_analyser import analyze
from CVM_indices.CVM_draft import analyze
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


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
        # Retrieve main Sobol indices
        sobol_values = sobol_indices[['S']].values.squeeze()
        sobol_total_values = sobol_indices[['ST']].values.squeeze()
        sobol_ind_values = sobol_indices[['S_ind']].values.squeeze()
        sobol_total_ind_values = sobol_indices[['ST_ind']].values.squeeze()
        fig, axs = plt.subplots(2, 2, figsize=(16, 16))
        # Display a piechart for Sobol indices
        axs[0, 0].set_title('Sobol')
        axs[0, 0].pie(sobol_values, labels=columns_names, autopct=label_pie(sobol_values), shadow=False, normalize=True)
        # Display a piechart for Sobol Total indices
        axs[0, 1].set_title('Sobol Total')
        axs[0, 1].pie(sobol_total_values, labels=columns_names, autopct=label_pie(sobol_total_values), shadow=False, normalize=True)
        # Display a piechart for Sobol independent indices
        axs[1, 0].set_title('Sobol independent')
        axs[1, 0].pie(sobol_ind_values, labels=columns_names, autopct=label_pie(sobol_ind_values), shadow=False, normalize=True)
        # Display a piechart for Sobol Total independent indices
        axs[1, 1].set_title('Sobol Total independent')
        axs[1, 1].pie(sobol_total_ind_values, labels=columns_names, autopct=label_pie(sobol_total_ind_values), shadow=False, normalize=True)
        plt.show()



class CVMIndices:
    
    @staticmethod
    def compute(dataframe, ouput):
        # Create a deep copy of the input dataframe (to not destroy/modify data)
        df = dataframe.copy(deep=True)
        # Concatenate the output column in the dataframe
        df['OUTPUT'] = ouput
        # Build the columns list and remove the output column
        list_of_columns = [[c] for c in df.columns]
        list_of_columns.remove(["OUTPUT"])
        # Add a defined epsilon to categorical columns
        eps = 1e-5
        # Handle categorical columns
        categorical_columns = ["Input_07"]
        for cat in categorical_columns:
            # extract the column and remove it from original dataset
            cat_serie = df[cat]
            df = df.drop(cat, axis=1)
            # perform one_hot encoding
            # take care to remove the first one hot encoded vector in order to avoid bias
            dum_cat_serie = pd.get_dummies(cat_serie, prefix="{}".format(cat), prefix_sep="=", drop_first=False)
            # add random to break ties
            dum_cat_serie = dum_cat_serie + pd.np.random.uniform(-eps, eps, dum_cat_serie.shape)
            # put it back in the dataframe
            df = pd.concat([df, dum_cat_serie], axis=1)
            # remove the column name and replace it with the list of newly created columns
            list_of_columns.remove([cat])
            list_of_columns.append(dum_cat_serie.columns.to_list())
        # Perform the computation of the CVM indices, with respect to the output column
        indices = analyze(df.sample(10000), 'OUTPUT', list_of_columns)
        return indices

    @staticmethod
    def visualize_barcharts(cvm_indices):
        # Retrieve xticks, cvm indices and cvm independent indices for further plotting
        columns = cvm_indices.index.tolist()
        cvm_series = cvm_indices['CVM'].tolist()
        cvm_independent_series = cvm_indices['CVM_indep'].tolist()
        # Compute xticks location and set the width of the bars
        x = np.arange(len(columns))
        width = 0.15
        # Initialize the plot
        fig, ax = plt.subplots(figsize=(13, 6))
        # Display both bars at each tick location for both indices
        rects1 = ax.bar(x - width/2, cvm_series, width, label='CVM')
        rects2 = ax.bar(x + width/2, cvm_independent_series, width, label='CVM independent')
        # Display a dedicated horizontal dashed line for 1.0 value (higher limit of indices)
        ax.plot([0, len(columns)-1], [1., 1.], '--r', linewidth=0.5)
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Index value')
        ax.set_title('CVM/CVM independent indices')
        ax.set_xticks(x)
        ax.set_xticklabels(columns)
        ax.set_ylim([0.0, 1.15])
        # Annotate values of CVM indices above bars
        for idx, cvm_value in enumerate(cvm_series):
            rounded_cvm = round(cvm_value, 3)
            ax.annotate(rounded_cvm, xy=(idx-0.14, cvm_value+0.05), rotation=90)
        # Annotate values of CVM independent indices above bars
        for idx, cvm_ind_value in enumerate(cvm_independent_series):
            rounded_cvm_ind = round(cvm_ind_value, 3)
            ax.annotate(rounded_cvm_ind, xy=(idx+0.05, cvm_ind_value+0.05), rotation=90)
        # Display the final plot with legend
        fig.tight_layout()
        ax.legend()
        plt.show()



