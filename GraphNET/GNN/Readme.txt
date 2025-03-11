Overview
This repository implements Graph Neural Networks without attention mechanisms for molecular property prediction tasks. The implementation focuses on Graph Convolutional Networks (GCN) for various prediction tasks.
Project Structure
Core Components (GCN folder)

    Graph Convolutional Network architecture
    Forward method implementations
    ECFP feature generators using molecular featurizers
    Core functionality imported by notebooks

Model Storage (Saved_models folder)

    Stores best-performing model parameters
    Automatic checkpoint management during training
    Parameters automatically cleaned after use
    Regular storage checks recommended

Main Notebooks
1. GCN_classification

    Purpose: Classification using basic GCN
    Input: CSV classification dataset
    Output: Dictionary containing:
        Host names
        Targets
        Raw predictions
        Classification thresholds
        Actual labels

2. GCN_Hyperparameter

    Purpose: Hyperparameter optimization
    Method: Random search on regression dataset
    Output: Best model configuration

3. GCN_regression

    Purpose: Log_D value prediction
    Input: CSV with log_D values per host/target
    Output: Dictionary containing:
        Host names
        Target names
        Predicted values
        Actual values

4. GCN_relative_regression

    Purpose: Relative property prediction
    Features:
        Custom model architecture within notebook
        Independent from GCN folder implementation
    Process:
        Processes log_D values
        Creates and featurizes host pairs
        Calculates relative values
    Output: Pairwise prediction dictionary
    Requires: Additional processing via data_processing notebook

Notes

    Core model modifications should be made in the GCN folder
    Regular checkpoint cleanup recommended
    Relative regression implementation is self-contained
