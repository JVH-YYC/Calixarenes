"""
Scikit-Learn functions for calixarene benchmarking
Random Forest and SVM using ECFP6 fingerprints"""

import sklearn as skl
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import MultiOutputClassifier
import Featurization.CalixSKLDatasets as CSD
import Featurization.calix_standard_settings as CSS
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

def create_single_split_ECFP_dataset(calixarene_csv_folder,
                                     calixarene_csv_file,
                                     target_columns,
                                     target_columns_per_example,
                                     split_method,
                                     train_fraction,
                                     test_fraction,
                                     peptide_one_hot_encoding):

    """
    A workflow that begins with a .csv file containing calixarene adsorption data,
    ends with a developed and split dataset for ML training using scikit-learn models.
    """

    # Load the data from the .csv file
    first_calix_dict = CSD.create_ecfp_dictionary(calixarene_csv_folder=calixarene_csv_folder,
                                                  calixarene_csv_file=calixarene_csv_file,
                                                  target_columns=target_columns,
                                                  target_columns_per_example=target_columns_per_example)
    
    split_calix_dict = CSD.split_calix_dataset(calixarene_dict=first_calix_dict,
                                               split_method=split_method,
                                               train_fraction=train_fraction,
                                               test_fraction=test_fraction)
    
    random_forest_input = CSD.organize_random_forest_input(split_calix_dataset=split_calix_dict,
                                                           dataset_target_type=target_columns_per_example,
                                                           ordered_feature_list=['ECFP'],
                                                           peptide_one_hot_encoding=peptide_one_hot_encoding)
    
    return random_forest_input

def train_single_random_forest(rfi):
    # Initialize the RandomForestRegressor
    rf = RandomForestRegressor(criterion='squared_error')
    
    # Train on the 'train' split
    rf.fit(rfi['train']['features'], rfi['train']['target'])

    # Evaluate on 'test' split
    predictions = rf.predict(rfi['test']['features'])
    mse = mean_squared_error(rfi['test']['target'], predictions)

    return mse

def perform_rf_grid_search(rfi,
                           mode,
                        plot_best_model=False,
                        save_pickle_file=False,
                        pickle_file_name='placeholder'):

    # Initialize the RandomForestRegressor
    if mode == 'regression':
        if len(rfi['train']['target'].shape) > 1:
            rf = MultiOutputRegressor(RandomForestRegressor(criterion='squared_error'))
            param_grid = {
                'estimator__n_estimators': [10, 25, 50, 100, 250,],
                'estimator__max_depth': [None, 10, 50],
                'estimator__min_samples_split': [2, 5, 10],
                'estimator__min_samples_leaf': [1, 2, 4],
                'estimator__bootstrap': [True,]}
        else:
            rf = RandomForestRegressor(criterion='squared_error')
            # Define the hyperparameters for grid search
            param_grid = {
                'n_estimators': [10, 25, 50, 100, 250,],
                'max_depth': [None, 10, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True,]}
    elif mode == 'classification':
        if len(rfi['train']['target'].shape) > 1:
            rf = MultiOutputClassifier(RandomForestClassifier(criterion='gini'))
            param_grid = {
                'estimator__n_estimators': [10, 25, 50, 100, 250,],
                'estimator__max_depth': [None, 10, 50],
                'estimator__min_samples_split': [2, 5, 10],
                'estimator__min_samples_leaf': [1, 2, 4],
                'estimator__bootstrap': [True,]}
        else:
            rf = RandomForestClassifier(criterion='gini')
            param_grid = {
                'n_estimators': [10, 25, 50, 100, 250,],
                'max_depth': [None, 10, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True,]}

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               scoring='neg_mean_squared_error', 
                               cv=10, 
                               verbose=1, 
                               n_jobs=-1)

    # Fit the model using the 'train' split
    grid_search.fit(rfi['train']['features'], rfi['train']['target'])

    # Predict on the 'test' split and compute MSE
    best_rf = grid_search.best_estimator_
    predictions_test = best_rf.predict(rfi['test']['features'])

    # Record the results
    results = pd.DataFrame(grid_search.cv_results_)

    # Sort the results by test score
    results = results.sort_values(by='mean_test_score', ascending=False)

    # Save the results to a CSV file
    results.to_csv('grid_search_results_sorted.csv', index=False)

   # Plot the best model's performance if plot_best_model is True
    if plot_best_model or save_pickle_file:
        if mode == 'regression':
            predictions_train = best_rf.predict(rfi['train']['features'])
        elif mode == 'classification':
            predictions_train = best_rf.predict_proba(rfi['train']['features'])
            if len(predictions_train) > 2:
                predictions_train = [prob[:, 1] for prob in predictions_train]
                train_combined = np.column_stack(predictions_train)
            predictions_test = best_rf.predict_proba(rfi['test']['features'])
            if len(predictions_test) > 2:
                predictions_test = [prob[:, 1] for prob in predictions_test]
                test_combined = np.column_stack(predictions_test)
        # Create list of tuples for train and test datasets
        train_data = list(zip(rfi['train']['target'], train_combined))
        test_data = list(zip(rfi['test']['target'], test_combined))
        
        # Save data to a pickle file if save_pickle_file is True
        if save_pickle_file:
            with open(pickle_file_name, 'wb') as f:
                pickle.dump({'train': train_data, 'test': test_data}, f)
        
        # Plot if plot_best_model is True
        if plot_best_model:
            plt.figure(figsize=(10, 6))
            plt.scatter(rfi['train']['target'], predictions_train, color='blue', label='Train', alpha=0.5)
            plt.scatter(rfi['test']['target'], predictions_test, color='orange', label='Test', alpha=0.5)
            
            # Draw the line of x=y
            # Get the current limits of the plot
            xlim = plt.gca().get_xlim()
            ylim = plt.gca().get_ylim()

            # Determine the min and max values for x=y line using the limits
            min_val = min(xlim[0], ylim[0])
            max_val = max(xlim[1], ylim[1])            
            plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='x=y line')

            # Setting the same scale for both axes
            plt.xlim(min_val, max_val)
            plt.ylim(min_val, max_val)
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs. Predicted Values for Best Model')
            plt.legend()
            plt.grid(True)
            plt.show()

    return grid_search.best_params_

def perform_svm_grid_search(svm_data,
                            mode,
                        plot_best_model=False,
                        save_pickle_file=False,
                        pickle_file_name='placeholder'):
    
    # Initialize the Support Vector Regressor and check for multiple outputs
    if len(svm_data['train']['target'].shape) > 1:
        # Define the hyperparameters for grid search
        param_grid = {
        'estimator__C': [0.1, 1, 10, 100, 1000],
        'estimator__kernel': ['rbf',],
        'estimator__gamma': ['scale', 'auto', 0.1, 1, 10]}
        if mode == 'regression':
            svm = MultiOutputRegressor(SVR())
            param_grid['estimator__epsilon']=[0.1, 0.2, 0.5, 1, 2, 5]

        elif mode == 'classification':
            svm = MultiOutputClassifier(SVC())
    else:
        # Define the hyperparameters for grid search
        param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'kernel': ['rbf',],
        'gamma': ['scale', 'auto', 0.1, 1, 10]}
        if mode == 'regression':
            svm = SVR()
            param_grid['epsilon']=[0.1, 0.2, 0.5, 1, 2, 5]
        elif mode == 'classification':
            svm = SVC()    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, 
                               scoring='neg_mean_squared_error', 
                               cv=10, 
                               verbose=1, 
                               n_jobs=-1)

    # Fit the model using the 'train' split
    grid_search.fit(svm_data['train']['features'], svm_data['train']['target'])

    # Predict on the 'test' split and compute MSE
    best_svm = grid_search.best_estimator_
    predictions_test = best_svm.predict(svm_data['test']['features'])

    # Record the results
    results = pd.DataFrame(grid_search.cv_results_)

    # Sort the results by test score
    results = results.sort_values(by='mean_test_score', ascending=False)

    # Save the results to a CSV file
    results.to_csv('grid_search_results_sorted.csv', index=False)

   # Plot the best model's performance if plot_best_model is True
    if plot_best_model or save_pickle_file:
        predictions_train = best_svm.predict(svm_data['train']['features'])

        # Create list of tuples for train and test datasets
        train_data = list(zip(svm_data['train']['target'], predictions_train))
        test_data = list(zip(svm_data['test']['target'], predictions_test))
        
        # Save data to a pickle file if save_pickle_file is True
        if save_pickle_file:
            with open(pickle_file_name, 'wb') as f:
                pickle.dump({'train': train_data, 'test': test_data}, f)
        
        # Plot if plot_best_model is True
        if plot_best_model:
            plt.figure(figsize=(10, 6))
            plt.scatter(svm_data['train']['target'], predictions_train, color='blue', label='Train', alpha=0.5)
            plt.scatter(svm_data['test']['target'], predictions_test, color='orange', label='Test', alpha=0.5)
            
            # Draw the line of x=y
            # Get the current limits of the plot
            xlim = plt.gca().get_xlim()
            ylim = plt.gca().get_ylim()

            # Determine the min and max values for x=y line using the limits
            min_val = min(xlim[0], ylim[0])
            max_val = max(xlim[1], ylim[1])            
            plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='x=y line')

            # Setting the same scale for both axes
            plt.xlim(min_val, max_val)
            plt.ylim(min_val, max_val)
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs. Predicted Values for Best Model')
            plt.legend()
            plt.grid(True)
            plt.show()

    return grid_search.best_params_


# rfi = create_single_split_ECFP_dataset('Featurization/',
#                                        'calix smiles absolute.csv',
#                                        ['H3K4me1',
#                                         'H3K4me2',
#                                         'H3K4me3',
#                                         'H3R2me2s',
#                                         'H3R2me2a',
#                                         'H3K9me3',
#                                         'H3K4ac'],
#                                        'each',
#                                        'by_point',
#                                        0.8,
#                                        0.1,
#                                        CSS.peptide_one_hot_encoding)

td = CSD.create_ecfp_dictionary(calixarene_csv_folder='Featurization/',
                                calixarene_csv_file='Categorical prediction data.csv',
                                target_columns=['H3K4me1',
                                                'H3K4me2',
                                                'H3K4me3',
                                                'H3R2me2s',
                                                'H3R2me2a',
                                                'H3K9me3',
                                                'H3K4ac',
                                                'H3K4'],
                                target_columns_per_example='each')

cv_sd = CSD.cross_validation_split_calix_dataset(calixarene_dict=td,
                                                 split_method='by_host',
                                                 train_fraction=0.8,
                                                 test_fraction=0.1,
                                                 num_folds=10)

# Usage:
for entry in range(10):
    curr_dict = cv_sd['CV' + str(entry)]
    rfi = CSD.organize_random_forest_input(split_calix_dataset=curr_dict,
                                           dataset_target_type='each',
                                           ordered_feature_list=['ECFP'],
                                           peptide_one_hot_encoding=CSS.peptide_one_hot_encoding)
    pickle_file_name = 'CV' + str(entry) + '_test_train_data.pkl'
    best_params = perform_rf_grid_search(rfi,
                                          mode = 'classification',
                                          plot_best_model=True,
                                          save_pickle_file=True,
                                          pickle_file_name=pickle_file_name)


