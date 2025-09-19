## Dataset

The project uses experimental binding data for **39 calixarene molecules** against **8 histone peptides**:

### Calixarene Families
- **Core A**: AP1-AP9, AM1-AM2, AH1-AH7, AO1-AO3
- **Core B**: BP0-BP1, BM1, BH2 
- **Core C**: CP1-CP2 
- **Core D**: DP2, DM1, DO2-DO3 
- **Core E**: E1, E3, E6-E7 
- **Core P**: PSC4, PNO2 

### Histone Peptides
- H3K4, H3K4ac, H3K4me1, H3K4me2, H3K4me3
- H3K9me3, H3R2me2a, H3R2me2s

## Machine Learning Approaches

### 1. Traditional Machine Learning (`SKLearnBenchmarks.py`)
- **Random Forest** and **Support Vector Machines**
- Uses **ECFP6 fingerprints** for molecular representation
- Supports both **regression** and **classification** tasks
- Implements **leave-one-out cross-validation** and **structured holdout** validation

### 2. Convolutional Neural Networks (`ResNet/`)
- **3D ResNet architecture** for processing molecular grids
- Processes **3D molecular representations** (48×48×48 grids)
- Supports both **absolute** and **relative** binding predictions
- Uses **PyTorch** with GPU acceleration

### 3. Graph Neural Networks (`GraphNET/`)
- **GCN (Graph Convolutional Networks)** and **AttentiveFP** architectures
- Processes **molecular graphs** directly
- Includes hyperparameter optimization notebooks
- Supports both regression and classification

## Project Structure

```
├── Alignment/           # Molecular alignment tools
├── ClusterData/         # Clustering analysis results
├── CSVFiles/           # Raw binding data and processed datasets
├── DataLoaders/        # Data loading utilities
├── Featurization/      # Molecular featurization (ECFP, SMILES)
├── GCN-AttentiveFP/          # Graph neural network implementations (Mainly contains python notebooks)
├── GridTools/         # 3D grid generation and processing
├── Hyperparam Search/ # Hyperparameter optimization results
├── ImportMol/         # Molecular structure import utilities
├── PQFiles/           # Parquet files with 3D molecular data
├── Raw Data/          # Raw experimental data
├── ResNet/            # CNN/ResNet implementations
├── Results Dictionaries/ # Model performance results
└── Visualization/    # Plotting and analysis tools
```

## Key Features

### Data Processing
- **SMILES to 3D structure** conversion using RDKit
- **Molecular alignment** for consistent 3D representations
- **ECFP6 fingerprint** generation for traditional ML
- **3D grid generation** for CNN approaches
- **Graph construction with** for GNN approaches

### Validation Strategies
- **Leave-One-Out (LOO)** cross-validation
- **Structured holdout** validation
- **10-fold cross-validation** for hyperparameter tuning
- **Train/validation/test** splits

### Model Evaluation
- **Regression metrics**: MSE, R², correlation coefficients
- **Classification metrics**: AUROC, precision, recall
- **Visualization**: Predicted vs actual plots, ROC curves
- **Statistical analysis**: Cross-validation results

## Usage Examples

### Traditional ML with Random Forest
```python
from SKLearnBenchmarks import loo_random_forest_final

# Leave-one-out cross-validation
results = loo_random_forest_final(
    calixarene_csv_folder='Featurization/',
    calixarene_csv_name='calix smiles absolute.csv',
    peptide_one_hot_encoding='one_hot_short.csv',
    calixarene_list=calixarene_list,
    output_name='RF_results.pkl',
    relative_training=False
)
```

### CNN Training
```python
from ResNet.CalixNet import cnn_training_split_workflow

# Train ResNet model
cnn_training_split_workflow(
    pq_file_directory='PQFiles/',
    pq_file_name='AlokThesis10A_Comb.pq',
    csv_file_directory='CSVFiles/',
    binding_file='Data excluding non-binders.csv',
    one_hot_file='one_hot_short.csv',
    split_calixarene_dict=split_dict,
    holdout_size=0.2,
    batch_size=400,
    training_epochs=400
)
```

### Data Visualization
```python
from CalixareneRegression import created_predicted_actual_regression_plot
from CalixareneAUROC import plot_multiple_auc_roc

# Plot regression results
created_predicted_actual_regression_plot(results_dict)

# Plot classification results  
plot_multiple_auc_roc(classification_results)
```

## Dependencies

### Core Libraries
- **Python 3.7+**
- **NumPy, Pandas** - Data manipulation
- **Scikit-learn** - Traditional ML algorithms
- **PyTorch** - Deep learning framework
- **RDKit** - Chemical informatics
- **Matplotlib, Seaborn** - Visualization

### Specialized Libraries
- **PyArrow** - Parquet file handling
- **UMAP** - Dimensionality reduction
- **TorchANI** - Neural network potentials

## Results Summary

The project demonstrates that:
1. **Traditional ML** (Random Forest, SVM) with ECFP fingerprints provides strong baseline performance
2. **Deep learning** approaches (CNN, GNN) can capture more complex molecular interactions
3. **Relative binding** predictions often outperform absolute predictions
4. **Leave-one-out** validation provides robust performance estimates

## File Descriptions

### Main Scripts
- `SKLearnBenchmarks.py` - Traditional ML implementations
- `ResNetTrain.py` - CNN training pipeline
- `CalixareneRegression.py` - Regression analysis and plotting
- `CalixareneAUROC.py` - Classification analysis and ROC curves
- `GridPopThesis.py` - 3D grid generation pipeline

### Data Files
- `CSVFiles/All binding data cleaned.csv` - Main experimental dataset
- `Featurization/calix smiles absolute.csv` - SMILES strings and binding data
- `PQFiles/` - 3D molecular grid data in Parquet format

