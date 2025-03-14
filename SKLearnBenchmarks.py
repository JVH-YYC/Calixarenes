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

def create_structured_ECFP_dataset(calixarene_csv_folder,
                                   calixarene_csv_file,
                                   peptide_one_hot_encoding,
                                   split_calixarene_dict,
                                   holdout_size,
                                   relative_training):
    """
    A function for final testing; to see how much the size of the test holdout influences performance,
    with both absolute and relative training approaches.

    LOO model input organization function will still work fine for this dataset split
    """

    if relative_training == True:
        first_calix_dict = CSD.create_structured_relative_ecfp_dictionary(calixarene_csv_folder=calixarene_csv_folder,
                                                                          calixarene_csv_file=calixarene_csv_file,
                                                                          split_calixarene_dict=split_calixarene_dict,
                                                                          holdout_size=holdout_size)
        model_input = CSD.organize_structured_relative_model_input(structured_calix_dataset=first_calix_dict,
                                                                      one_hot_encoding_folder=calixarene_csv_folder,
                                                                      peptide_one_hot_encoding=peptide_one_hot_encoding)
        
    else:
        first_calix_dict = CSD.create_structured_absolute_ecfp_dictionary(calixarene_csv_folder=calixarene_csv_folder,
                                                                          calixarene_csv_file=calixarene_csv_file,
                                                                          split_calixarene_dict=split_calixarene_dict,
                                                                          holdout_size=holdout_size)
        
        model_input = CSD.organize_structured_absolute_model_input(structured_calix_dataset=first_calix_dict,
                                                                      one_hot_encoding_folder=calixarene_csv_folder,
                                                                      peptide_one_hot_encoding=peptide_one_hot_encoding)
    
    return model_input

def create_LOO_absolute_datasets(calixarene_csv_folder,
                                 calixarene_csv_file,
                                 peptide_one_hot_encoding,
                                 holdout_calixarene):
    """
    A function related to that directly above, to create leave-one-out datasets for calixarene adsorption data.
    In this case, absolute target values are used.
    """

    # Load the data from the .csv file
    first_calix_dict = CSD.create_loo_ecfp_dictionary(calixarene_csv_folder=calixarene_csv_folder,
                                                      calixarene_csv_file=calixarene_csv_file,
                                                      holdout_calixarene=holdout_calixarene)
    
    absolute_model_input, peptide_name_list = CSD.organize_loo_model_input(loo_calix_dataset=first_calix_dict,
                                                       one_hot_encoding_folder=calixarene_csv_folder,
                                                       peptide_one_hot_encoding=peptide_one_hot_encoding,
                                                       relative_training=False)
    
    return absolute_model_input, peptide_name_list

def create_LOO_relative_datasets(calixarene_csv_folder,
                                 calixarene_csv_file,
                                 peptide_one_hot_encoding,
                                 holdout_calixarene,
                                 method):
    """
    A function related to that directly above, to create leave-one-out datasets for calixarene adsorption data.
    In this case, relative target values are used.
    """

    # Load the data from the .csv file
    first_calix_dict = CSD.create_loo_relative_ecfp_dictionary(calixarene_csv_folder=calixarene_csv_folder,
                                                               calixarene_csv_file=calixarene_csv_file,
                                                               holdout_calixarene=holdout_calixarene,
                                                               method=method)
    
    relative_model_input, peptide_name_list = CSD.organize_loo_model_input(loo_calix_dataset=first_calix_dict,
                                                       one_hot_encoding_folder=calixarene_csv_folder,
                                                       peptide_one_hot_encoding=peptide_one_hot_encoding,
                                                       relative_training=True)
    
    return relative_model_input, peptide_name_list

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
            train_combined = best_rf.predict(rfi['train']['features'])
            test_combined = predictions_test
        elif mode == 'classification':
            predictions_train = best_rf.predict_proba(rfi['train']['features'])
            if type(predictions_train)==list:
                predictions_train = [prob[:, 1] for prob in predictions_train]
                train_combined = np.column_stack(predictions_train)
            else:
                train_combined = predictions_train[:, 1]
            predictions_test = best_rf.predict_proba(rfi['test']['features'])
            if type(predictions_test)==list:
                predictions_test = [prob[:, 1] for prob in predictions_test]
                test_combined = np.column_stack(predictions_test)
            else:
                test_combined = predictions_test[:, 1]
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
            plt.scatter(rfi['train']['target'], train_combined, color='blue', label='Train', alpha=0.5)
            plt.scatter(rfi['test']['target'], test_combined, color='orange', label='Test', alpha=0.5)
            
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
    multi_class = False
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
            multi_class = True
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
    print(svm_data['train']['target'])
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
        if mode == 'regression':
            train_combined = best_svm.predict(svm_data['train']['features'])
            test_combined = predictions_test
        elif mode == 'classification':
            if multi_class == False:
                train_combined = best_svm.decision_function(svm_data['train']['features'])
                test_combined = best_svm.decision_function(svm_data['test']['features'])
            else:
                train_scores = [estimator.decision_function(svm_data['train']['features']) for estimator in best_svm.estimators_]
                train_combined = np.column_stack(train_scores)

                test_scores = [estimator.decision_function(svm_data['test']['features']) for estimator in best_svm.estimators_]
                test_combined = np.column_stack(test_scores)

        # Create list of tuples for train and test datasets
        train_data = list(zip(svm_data['train']['target'], train_combined))
        test_data = list(zip(svm_data['test']['target'], test_combined))
        
        # Save data to a pickle file if save_pickle_file is True
        if save_pickle_file:
            with open(pickle_file_name, 'wb') as f:
                pickle.dump({'train': train_data, 'test': test_data}, f)
        
        # Plot if plot_best_model is True
        if plot_best_model:
            plt.figure(figsize=(10, 6))
            plt.scatter(svm_data['train']['target'], train_combined, color='blue', label='Train', alpha=0.5)
            plt.scatter(svm_data['test']['target'], test_combined, color='orange', label='Test', alpha=0.5)
            
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

def loo_random_forest_final(calixarene_csv_folder,
                                calixarene_csv_name,
                                peptide_one_hot_encoding,
                                calixarene_list,
                                output_name,
                                relative_training,
                                method,
                                n_estimators=100,
                                max_depth=10,
                                min_samples_split=2,
                                min_samples_leaf=4,
                                bootstrap=True):
    """
    A function to perform the final leave-one-out cross validation for random forest. With no need for early stopping/paramater tuning,
    all non-held-out calixarenes are used as the training set.

    Saves predicted/actual results in a dictionary of the typical format, and pickles it for future plotting/processing.

    Results from 10-fold-CV hyperparameter search:
    Best = Bootstrap Tue, Max_depth 10, min_samples_leaf 4, min_samples_split 2, n_estimators 100
    Worst = Bootstrap False, Max_depth 50, min_samples_leaf 1, min_samples_split 2, n_estimators 10

    """

    # Initialize the dictionary to hold the results
    loo_results = {}
    loo_int_results = {}
    # Loop through each calixarene
    for calix in calixarene_list:
        # Create the dataset
        if relative_training is not None:
            rfi, peptide_name_list = create_LOO_relative_datasets(calixarene_csv_folder=calixarene_csv_folder,
                                               calixarene_csv_file=calixarene_csv_name,
                                               peptide_one_hot_encoding=peptide_one_hot_encoding,
                                               holdout_calixarene=calix,
                                               method=method)
            # Initialize the Random Forest Regressor
            rfr = RandomForestRegressor(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        bootstrap=bootstrap)

            # Train on the 'train' split
            rfr.fit(rfi['train']['features'], rfi['train']['target'])

            # Evaluate on 'test' split. Must re-shape the features, as it's a single sample.
            predictions = rfr.predict(rfi['test']['features'])
            mse = mean_squared_error(rfi['test']['target'], predictions)

            #Organize lists
            actual_values = rfi['test']['target']
            predicted_diffs = predictions
            test_calix_positions = rfi['test']['test_pos']
            peptide_names = rfi['test']['peptide_order']
            known_calix_values = rfi['test']['known_val']

            # Create lists for storing intermediate results
            loo_int_results[calix] = {name: {'actual': [], 'predicted': []} for name in peptide_name_list}
            for actual, predicted_diff, position, peptide_name, known_calix in zip(
                    actual_values, predicted_diffs, test_calix_positions, peptide_names, known_calix_values):
                
                # Calculate the predicted value for the unknown calix
                if position == 'row1':
                    predicted_value = predicted_diff + known_calix
                    act_val = actual + known_calix  
                else:
                    predicted_value = -1 * (predicted_diff - known_calix)
                    act_val = -1 * (actual - known_calix) 

                # Append the actual and predicted values to the loo_int_results dictionary
                loo_int_results[calix][peptide_name]['actual'].append(act_val)
                loo_int_results[calix][peptide_name]['predicted'].append(predicted_value)

            loo_results[calix] = {name: {'actual': np.mean(loo_int_results[calix][name]['actual']),
                                               'predicted': np.mean(loo_int_results[calix][name]['predicted'])} for name in peptide_name_list}

     
        else:
            rfi, peptide_name_list = create_LOO_absolute_datasets(calixarene_csv_folder=calixarene_csv_folder,
                                                                  calixarene_csv_file=calixarene_csv_name,
                                                                  peptide_one_hot_encoding=peptide_one_hot_encoding,
                                                                  holdout_calixarene=calix)

            # Initialize the RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    bootstrap=bootstrap)

            # Train on the 'train' split
            rf.fit(rfi['train']['features'], rfi['train']['target'])

            # Evaluate on 'test' split. Must re-shape the features, as it's a single sample.
            predictions = rf.predict(rfi['test']['features'])
            mse = mean_squared_error(rfi['test']['target'], predictions)
            # Save the results to the dictionary
            loo_results[calix] = {name: {'actual': rfi['test']['target'][i], 'predicted': predictions[i]} for i, name in enumerate(peptide_name_list)}

    # Save the dictionary to a pickle file
    with open(output_name, 'wb') as f:
        pickle.dump(loo_results, f)

    return loo_results

def loo_svm_final(calixarene_csv_folder,
                                calixarene_csv_name,
                                peptide_one_hot_encoding,
                                calixarene_list,
                                output_name,
                                relative_training,
                                method,
                                C=100,
                                kernel='rbf',
                                gamma='scale',
                                epsilon=0.1):
    """
    An equivalent function to the random forest function above. After 10-fold CV hyperparameter searching, the appropriate parameters were found to be:
    C = 100, epsilon = 0.1, kernel = rbf, gamma = scale
    """
    # Initialize the dictionary to hold the results
    loo_int_results = {}
    loo_results = {}

    # Loop through each calixarene
    for calix in calixarene_list:
        # Create the dataset
        if relative_training:
            svi, peptide_name_list = create_LOO_relative_datasets(calixarene_csv_folder=calixarene_csv_folder,
                                                                  calixarene_csv_file=calixarene_csv_name,
                                                                  peptide_one_hot_encoding=peptide_one_hot_encoding,
                                                                  holdout_calixarene=calix,
                                                                  method=method)
            # Initialize the Support Vector Regressor
            svm = SVR(C=C,
                    kernel=kernel,
                    gamma=gamma,
                    epsilon=epsilon)

            # Train on the 'train' split
            svm.fit(svi['train']['features'], svi['train']['target'])

            # Evaluate on 'test' split. Must re-shape the features, as it's a single sample.
            predictions = svm.predict(svi['test']['features'])
            mse = mean_squared_error(svi['test']['target'], predictions)

            #Organize lists
            actual_values = svi['test']['target']
            predicted_diffs = predictions
            test_calix_positions = svi['test']['test_pos']
            peptide_names = svi['test']['peptide_order']
            known_calix_values = svi['test']['known_val']

            # Create lists for storing intermediate results
            loo_int_results[calix] = {name: {'actual': [], 'predicted': []} for name in peptide_name_list}
            for actual, predicted_diff, position, peptide_name, known_calix in zip(
                    actual_values, predicted_diffs, test_calix_positions, peptide_names, known_calix_values):
                
                # Calculate the predicted value for the unknown calix
                if position == 'row1':
                    predicted_value = predicted_diff + known_calix
                    act_val = actual + known_calix  
                else:
                    predicted_value = -1 * (predicted_diff - known_calix)
                    act_val = -1 * (actual - known_calix) 

                # Append the actual and predicted values to the loo_int_results dictionary
                loo_int_results[calix][peptide_name]['actual'].append(act_val)
                loo_int_results[calix][peptide_name]['predicted'].append(predicted_value)

            loo_results[calix] = {name: {'actual': np.mean(loo_int_results[calix][name]['actual']),
                                               'predicted': np.mean(loo_int_results[calix][name]['predicted'])} for name in peptide_name_list}

        else:
            svi, peptide_name_list = create_LOO_absolute_datasets(calixarene_csv_folder=calixarene_csv_folder,
                                                                  calixarene_csv_file=calixarene_csv_name,
                                                                  peptide_one_hot_encoding=peptide_one_hot_encoding,
                                                                  holdout_calixarene=calix)

            # Initialize the Support Vector Regressor
            svm = SVR(C=C,
                    kernel=kernel,
                    gamma=gamma,
                    epsilon=epsilon)

            # Train on the 'train' split
            svm.fit(svi['train']['features'], svi['train']['target'])

            # Evaluate on 'test' split. Must re-shape the features, as it's a single sample.
            predictions = svm.predict(svi['test']['features'])
            mse = mean_squared_error(svi['test']['target'], predictions)
            # Save the results to the dictionary
            loo_results[calix] = {name: {'actual': svi['test']['target'][i], 'predicted': predictions[i]} for i, name in enumerate(peptide_name_list)}

    # Save the dictionary to a pickle file
    with open(output_name, 'wb') as f:
        pickle.dump(loo_results, f)

    return loo_results

def svm_structured_final(calixarene_csv_folder,
                         calixarene_csv_file,
                         peptide_one_hot_encoding,
                         holdout_size,
                         num_trials,
                         relative_training,
                         split_calixarene_dict,
                         output_name,
                         C=100,
                         kernel='rbf',
                         gamma='scale',
                         epsilon=0.1):
    """
    A function set up to test holding out different amounts of training data from SVM model.

    This is the last thing done - some hyperparameter optimization, 'concat' vs 'diff' method, etc. have all been established

    """                                

    # Initialize the dictionary to hold the results
    split_int_results = {}
    split_results = {}

    # Open copy of one_hot_encoding, to concatenate test items at prediction time
    
    # Loop through each repeat trial
    for repeat in range(num_trials):
        # Create the dataset
        if relative_training:
            svi, peptide_name_list = create_structured_ECFP_dataset(calixarene_csv_folder=calixarene_csv_folder,
                                                                    calixarene_csv_file=calixarene_csv_file,
                                                                    peptide_one_hot_encoding=peptide_one_hot_encoding,
                                                                    split_calixarene_dict=split_calixarene_dict,
                                                                    holdout_size=holdout_size,
                                                                    relative_training=True)
            
            # Initialize the Support Vector Regressor
            svm = SVR(C=C,
                    kernel=kernel,
                    gamma=gamma,
                    epsilon=epsilon)

            # Train on the 'train' split
            svm.fit(svi['train']['features'], svi['train']['target'])

            # Evaluate on 'test' split. Must re-shape the features, as it's a single sample.
            predictions = svm.predict(svi['test']['features'])
            mse = mean_squared_error(svi['test']['target'], predictions)

            #Organize lists
            actual_values = svi['test']['target']
            predicted_diffs = predictions
            test_calix_positions = svi['test']['test_pos']
            peptide_names = svi['test']['peptide_order']
            known_calix_values = svi['test']['known_val']

            # Create lists for storing intermediate results
            split_int_results[calix] = {name: {'actual': [], 'predicted': []} for name in peptide_name_list}
            for actual, predicted_diff, position, peptide_name, known_calix in zip(
                    actual_values, predicted_diffs, test_calix_positions, peptide_names, known_calix_values):
                
                # Calculate the predicted value for the unknown calix
                if position == 'row1':
                    predicted_value = predicted_diff + known_calix
                    act_val = actual + known_calix  
                else:
                    predicted_value = -1 * (predicted_diff - known_calix)
                    act_val = -1 * (actual - known_calix) 

                # Append the actual and predicted values to the loo_int_results dictionary
                split_int_results[calix][peptide_name]['actual'].append(act_val)
                split_int_results[calix][peptide_name]['predicted'].append(predicted_value)

            split_results[calix] = {name: {'actual': np.mean(split_int_results[calix][name]['actual']),
                                               'predicted': np.mean(split_int_results[calix][name]['predicted'])} for name in peptide_name_list}

        else:
            svi, peptide_name_list = create_structured_ECFP_dataset(calixarene_csv_folder=calixarene_csv_folder,
                                                                    calixarene_csv_file=calixarene_csv_file,
                                                                    peptide_one_hot_encoding=peptide_one_hot_encoding,
                                                                    split_calixarene_dict=split_calixarene_dict,
                                                                    holdout_size=holdout_size,
                                                                    relative_training=False)

            # Initialize the Support Vector Regressor
            svm = SVR(C=C,
                    kernel=kernel,
                    gamma=gamma,
                    epsilon=epsilon)

            # Train on the 'train' split
            svm.fit(svi['train']['features'], svi['train']['target'])

            split_results[str(repeat)] = {}
            for entry in svi['test']:
                if entry.split('_')[0] not in split_results[str(repeat)]:
                    split_results[str(repeat)][entry.split('_')[0]] = []
                test_entry = np.concatenate((svi['test'][entry]['ECFP'], svi['test'][entry]['Peptide_OH']), axis=0)
                curr_pred = svm.predict(test_entry.reshape(1, -1))
                split_results[str(repeat)][entry.split('_')[0]].append((curr_pred[0], svi['test'][entry]['Target_Val']))

    # Save the dictionary to a pickle file
    with open(output_name, 'wb') as f:
        pickle.dump(split_results, f)

    return split_results

def rf_structured_final(calixarene_csv_folder,
                        calixarene_csv_file,
                        peptide_one_hot_encoding,
                        holdout_size,
                        num_trials,
                        relative_training,
                        split_calixarene_dict,
                        output_name,
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=2,
                        min_samples_leaf=4,
                        bootstrap=True):
    """
    A function set up to test holding out different amounts of training data from SVM model.

    This is the last thing done - some hyperparameter optimization, 'concat' vs 'diff' method, etc. have all been established

    """                                

    # Initialize the dictionary to hold the results
    split_results = {}

    # For this function, always evaluate every peptide
    peptide_name_list = ['H3K4', 'H3K4ac', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9me3', 'H3R2me2a', 'H3R2me2s']

    # Loop through each repeat trial
    for repeat in range(num_trials):
        # Create the dataset
        if relative_training:
            split_results[str(repeat)] = {}
            split_int_results = {}

            rfi = create_structured_ECFP_dataset(calixarene_csv_folder=calixarene_csv_folder,
                                                 calixarene_csv_file=calixarene_csv_file,
                                                 peptide_one_hot_encoding=peptide_one_hot_encoding,
                                                 split_calixarene_dict=split_calixarene_dict,
                                                 holdout_size=holdout_size,
                                                 relative_training=True)
            
            # Initialize the Support Vector Regressor
            rfr = RandomForestRegressor(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        bootstrap=bootstrap)

            # Train on the 'train' split
            rfr.fit(rfi['train']['features'], rfi['train']['target'])

            # Evaluate on test split, one sample at a time
            holdout_calix = rfi['holdout']
            important_calix = [calix for calix in holdout_calix if calix.split('_')[0][0] in ['A', 'E', 'P']]
            # Create overall list for each calix
            # Organize lists. Each entry is a 3 position tuple (calix_1, calix_2, peptide)
            for entry in rfi['test']:
                if entry[0] in important_calix:
                    curr_calix = entry[0]
                elif entry[1] in important_calix:
                    curr_calix = entry[1]
                else:
                    continue

                test_entry = np.concatenate((rfi['test'][entry]['ECFP'], rfi['test'][entry]['Peptide_OH']), axis=0)
                predicted_diff = rfr.predict(test_entry.reshape(1, -1))
                actual_value = rfi['test'][entry]['Target_Val']
                test_calix_position = rfi['test'][entry]['test_pos']
                peptide_name = entry[2]
                known_calix_value = rfi['test'][entry]['known_val']

                # Create lists for storing intermediate results
                if curr_calix not in split_int_results:
                    split_int_results[curr_calix] = {name: {'actual': [], 'predicted': []} for name in peptide_name_list}
                  
                # Calculate the predicted value for the unknown calix
                if test_calix_position == 'row1':
                    predicted_value = predicted_diff + known_calix_value
                    act_val = actual_value + known_calix_value  
                else:
                    predicted_value = -1 * (predicted_diff - known_calix_value)
                    act_val = -1 * (actual_value - known_calix_value) 

                # Append the actual and predicted values to the loo_int_results dictionary
                split_int_results[curr_calix][peptide_name]['actual'].append(act_val)
                split_int_results[curr_calix][peptide_name]['predicted'].append(predicted_value)

            for measured_calix in important_calix:
                split_results[str(repeat)][measured_calix] = [(np.mean(split_int_results[measured_calix][name]['predicted']), np.mean(split_int_results[measured_calix][name]['actual'])) for name in peptide_name_list]

        else:
            rfi = create_structured_ECFP_dataset(calixarene_csv_folder=calixarene_csv_folder,
                                                 calixarene_csv_file=calixarene_csv_file,
                                                 peptide_one_hot_encoding=peptide_one_hot_encoding,
                                                 split_calixarene_dict=split_calixarene_dict,
                                                 holdout_size=holdout_size,
                                                 relative_training=False)

            # Initialize the Support Vector Regressor
            rfr = RandomForestRegressor(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        bootstrap=bootstrap)

            # Train on the 'train' split
            rfr.fit(rfi['train']['features'],rfi['train']['target'])

            # Only predict key calixarenes for comparison to LOO results
            split_results[str(repeat)] = {}
            for entry in rfi['test']:
                if entry.split('_')[0][0] in ['A', 'E', 'P']:
                    if entry.split('_')[0] not in split_results[str(repeat)]:
                        split_results[str(repeat)][entry.split('_')[0]] = []
                    test_entry = np.concatenate((rfi['test'][entry]['ECFP'], rfi['test'][entry]['Peptide_OH']), axis=0)
                    curr_pred = rfr.predict(test_entry.reshape(1, -1))
                    split_results[str(repeat)][entry.split('_')[0]].append((curr_pred[0], rfi['test'][entry]['Target_Val']))

    # Save the dictionary to a pickle file
    with open(output_name, 'wb') as f:
        pickle.dump(split_results, f)

    return split_results

split_calix_dict = {'predictable': ['AP1', 'AP3', 'AP4', 'AP5', 'AP6',
                                    'AP7', 'AP8', 'AP9', 'AH1', 'AH2',
                                    'AH3', 'AH4', 'AH5', 'AH6', 'AH7',
                                    'AM1', 'AM2', 'AO1', 'AO2', 'AO3',
                                    'E1', 'E3', 'E6', 'E7', 'E8', 'E11',
                                    'P-NO2', 'PSC4'],
                    'unpredictable': ['BP0', 'BP1', 'BH2', 'BM1', 'CP1',
                                      'CP2', 'DP2', 'DM1', 'DO2', 'DO3']}

# calixarene_list = ['AP1', 'AP3', 'AP4', 'AP5', 'AP6',
#                    'AP7', 'AP8', 'AP9', 'AH1', 'AH2',
#                    'AH3', 'AH4', 'AH5', 'AH6', 'AH7',
#                    'AM1', 'AM2', 'AO1', 'AO2', 'AO3',
#                    'BP0', 'BP1', 'BH2', 'BM1', 'CP1',
#                    'CP2', 'DP2', 'DM1', 'DO2', 'DO2', 'DO3',
#                    'E1', 'E3', 'E6', 'E7', 'E8', 'E11',
#                    'P-NO2', 'PSC4']

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

# td = CSD.create_relative_ecfp_dictionary(calixarene_csv_folder='Featurization/',
#                                 calixarene_csv_file='calix smiles absolute.csv',
#                                 target_columns=['H3K4me1',
#                                                 'H3K4me2',
#                                                 'H3K4me3',
#                                                 'H3R2me2s',
#                                                 'H3R2me2a',
#                                                 'H3K9me3',
#                                                 'H3K4ac',
#                                                 'H3K4'],
#                                 target_columns_per_example='all')

# cv_sd = CSD.cross_validation_split_calix_dataset(calixarene_dict=td,
#                                                  split_method='by_host',
#                                                  train_fraction=0.8,
#                                                  test_fraction=0.1,
#                                                  num_folds=10)

# Usage:
# for entry in range(10):
#     curr_dict = cv_sd['CV' + str(entry)]
#     rfi = CSD.organize_random_forest_input(split_calix_dataset=curr_dict,
#                                            dataset_target_type='all',
#                                            ordered_feature_list=['ECFP'],
#                                            peptide_one_hot_encoding=CSS.peptide_one_hot_encoding)
#     pickle_file_name = 'CV' + str(entry) + 'regression_svm_first_rel_test.pkl'
#     best_params = perform_svm_grid_search(rfi,
#                                           mode = 'regression',
#                                           plot_best_model=True,
#                                           save_pickle_file=True,
#                                           pickle_file_name=pickle_file_name)


