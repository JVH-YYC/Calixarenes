Overview
This repository contains an implementation of a Graph Neural Network (GNN) with attention mechanism, based on the AttentiveFP architecture. The codebase is designed for molecular property prediction tasks using SMILES representations.
Project Structure
Core Components (code folder)

    AttentiveFP: Contains the main GNN architecture implementation
        Graph Convolutional Network (GCN_FP) implementation
        Forward method definitions
        Molecular featurizers using RDKit for ECFP feature generation

Model Checkpoints
The system automatically saves model parameters during training in dedicated folders. Best performing models are preserved for later use in testing.
Main Notebooks
1. AttentionFP_Classification

    Purpose: Binary classification tasks using attention mechanism
    Input: CSV files from data folder
    Output: Pickle files containing:
        Raw predictions
        Classification thresholds
        Actual labels

2. AttentiveFP_Hyper

    Purpose: Hyperparameter optimization via random search
    Application: Results used across all model variants (classification, regression, and relative neural networks)

3. AttentiveFP_regression

    Purpose: Regression analysis for log_D values
    Input: CSV files with log_D values per target
    Output: Dictionary containing:
        Host names
        Targets
        Predicted values
        Actual values
    Results saved in pickle format

4. Relative_FP

    Purpose: Relative training for regression problems
    Input: CSV files with log_D values
    Process:
        Pairs hosts and calculates relative values
        Uses double input features (two SMILES per pair)
    Output: Predictions for paired compounds
    Companion notebook: Data processing for individual host predictions

License
Original license from the source paper is included in the LICENSE file.
Notes

    Model checkpoints are managed automatically during training
    Debug if checkpoint files aren't being cleaned up properly
    Core architecture modifications should be made in the code folder
