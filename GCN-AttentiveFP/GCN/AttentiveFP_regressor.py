"""
AttentiveFP Regressor - Clean implementation for molecular property prediction.

Uses the AttentiveFP model (Fingerprint class from AttentiveLayers.py) which implements
attention mechanisms at both atom and molecule levels for learning molecular representations.

Key features:
- Atom-level attention: learns which neighboring atoms are most important
- Molecule-level attention: learns which atoms contribute most to the molecular fingerprint
- GRU cells for iterative feature updates

Usage:
    python AttentiveFP_regressor.py --data_path ../Database/data.csv --targets H3K4 H3K4me3
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from rdkit import Chem

# Import existing modules - Fingerprint is the AttentiveFP model with attention mechanism
from GCN import Fingerprint, get_smiles_dicts, get_smiles_array


# =============================================================================
# 2. CONFIGURATION
# =============================================================================
@dataclass
class Config:
    """Configuration for AttentiveFP training and model architecture."""

    # Data parameters
    data_path: str = "../Database/calix smiles small set.csv"
    smiles_column: str = "SMILES"
    host_column: str = "Host"
    targets: List[str] = field(default_factory=lambda: [
        'H3K4', 'H3K4ac', 'H3K4me1', 'H3K4me2',
        'H3K4me3', 'H3K9me3', 'H3R2me2a', 'H3R2me2s'
    ])

    # Model architecture
    fingerprint_dim: int = 150
    radius: int = 3
    T: int = 2
    p_dropout: float = 0.1

    # Training parameters
    batch_size: int = 38
    num_epochs: int = 800
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    patience: int = 30
    min_delta: float = 0.001

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Output
    save_model: bool = True
    model_path: str = "attentivefp_model.pt"


# =============================================================================
# 3. TRAINER CLASS
# =============================================================================
class Trainer:
    """Handles training, evaluation, and cross-validation of AttentiveFP model."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.feature_dicts: Optional[Dict] = None

    def prepare_data(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare molecular features from SMILES.

        Returns:
            Filtered dataframe and feature dictionaries.
        """
        smiles_list = dataframe[self.config.smiles_column].values
        print(f"Number of input SMILES: {len(smiles_list)}")

        # Process and validate SMILES
        valid_smiles = []
        canonical_smiles = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_smiles.append(smiles)
                    canonical_smiles.append(Chem.MolToSmiles(mol, isomericSmiles=True))
            except Exception as e:
                print(f"Failed to process SMILES: {smiles}")
                continue

        print(f"Successfully processed SMILES: {len(valid_smiles)}")

        # Filter dataframe
        df = dataframe[dataframe[self.config.smiles_column].isin(valid_smiles)].copy()
        df['cano_smiles'] = canonical_smiles

        # Generate features using existing GCN function
        self.feature_dicts = get_smiles_dicts(valid_smiles)

        # Filter by available features
        df = df[df['cano_smiles'].isin(self.feature_dicts['smiles_to_atom_mask'].keys())]

        return df, self.feature_dicts

    def create_model(self, num_atom_features: int, num_bond_features: int) -> Fingerprint:
        """
        Create a new AttentiveFP model instance.

        The Fingerprint class implements AttentiveFP with:
        - Atom-level attention over neighbors (radius iterations)
        - Molecule-level attention over atoms (T iterations)
        """
        model = Fingerprint(
            radius=self.config.radius,
            T=self.config.T,
            input_feature_dim=num_atom_features,
            input_bond_dim=num_bond_features,
            fingerprint_dim=self.config.fingerprint_dim,
            output_units_num=len(self.config.targets),
            p_dropout=self.config.p_dropout
        )
        return model.to(self.device)

    def _get_batches(self, dataset: pd.DataFrame, shuffle: bool = True) -> List[np.ndarray]:
        """Create batches from dataset indices."""
        n_samples = len(dataset)
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        if n_samples <= self.config.batch_size:
            return [dataset.index.values]

        batches = []
        for i in range(0, n_samples, self.config.batch_size):
            batch_idx = indices[i:i + self.config.batch_size]
            batches.append(dataset.index.values[batch_idx])
        return batches

    def _prepare_batch(self, batch_df: pd.DataFrame) -> Tuple[torch.Tensor, ...]:
        """Convert batch dataframe to tensors using existing get_smiles_array."""
        smiles_list = batch_df.cano_smiles.values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, _ = get_smiles_array(
            smiles_list, self.feature_dicts
        )

        return (
            torch.Tensor(x_atom).to(self.device),
            torch.Tensor(x_bonds).to(self.device),
            torch.LongTensor(x_atom_index).to(self.device),
            torch.LongTensor(x_bond_index).to(self.device),
            torch.Tensor(x_mask).to(self.device)
        )

    def train_epoch(self, model: Fingerprint, dataset: pd.DataFrame,
                    optimizer: optim.Optimizer, loss_fn: nn.Module,
                    ratio_list: List[float] = None) -> float:
        """Train for one epoch."""
        model.train()

        if ratio_list is None:
            ratio_list = [1.0] * len(self.config.targets)

        batches = self._get_batches(dataset, shuffle=True)
        total_loss = 0.0

        for batch_indices in batches:
            batch_df = dataset.loc[batch_indices]
            x_atom, x_bonds, x_atom_index, x_bond_index, x_mask = self._prepare_batch(batch_df)

            optimizer.zero_grad()
            _, mol_prediction = model(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask)

            # Calculate weighted loss per target
            loss = 0.0
            for i, target in enumerate(self.config.targets):
                y_pred = mol_prediction[:, i]
                y_true = torch.Tensor(batch_df[target].values).squeeze().to(self.device)
                target_loss = loss_fn(y_pred, y_true) * (ratio_list[i] ** 2)
                loss += target_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(batches)

    @torch.no_grad()
    def evaluate(self, model: Fingerprint, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model on a dataset."""
        model.eval()

        batches = self._get_batches(dataset, shuffle=False)

        # Containers for metrics
        eval_MAE = {t: [] for t in self.config.targets}
        eval_MSE = {t: [] for t in self.config.targets}
        y_pred_all = {t: [] for t in self.config.targets}
        y_true_all = {t: [] for t in self.config.targets}
        smiles_list = []

        for batch_indices in batches:
            batch_df = dataset.loc[batch_indices]
            smiles_list.extend(batch_df.cano_smiles.values)

            x_atom, x_bonds, x_atom_index, x_bond_index, x_mask = self._prepare_batch(batch_df)
            _, mol_prediction = model(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask)

            for i, target in enumerate(self.config.targets):
                y_pred = mol_prediction[:, i].cpu().numpy()
                y_true = batch_df[target].values

                y_pred_all[target].extend(y_pred)
                y_true_all[target].extend(y_true)

                eval_MAE[target].extend(np.abs(y_pred - y_true))
                eval_MSE[target].extend((y_pred - y_true) ** 2)

        # Aggregate metrics
        mae = np.array([np.mean(eval_MAE[t]) for t in self.config.targets])
        mse = np.array([np.mean(eval_MSE[t]) for t in self.config.targets])

        return {
            'mae': mae,
            'mse': mse,
            'mean_mae': mae.mean(),
            'mean_mse': mse.mean(),
            'predictions': y_pred_all,
            'actuals': y_true_all,
            'smiles': smiles_list
        }

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None,
            model: Fingerprint = None) -> Tuple[Fingerprint, Dict]:
        """
        Train the model with optional validation and early stopping.

        Returns:
            Trained model and training history.
        """
        # Get feature dimensions
        sample_smiles = [train_df.cano_smiles.values[0]]
        x_atom, x_bonds, _, _, _, _ = get_smiles_array(sample_smiles, self.feature_dicts)
        num_atom_features = x_atom.shape[-1]
        num_bond_features = x_bonds.shape[-1]

        if model is None:
            model = self.create_model(num_atom_features, num_bond_features)

        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        loss_fn = nn.MSELoss()

        # Training state
        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(model, train_df, optimizer, loss_fn)
            history['train_loss'].append(train_loss)

            if val_df is not None:
                val_metrics = self.evaluate(model, val_df)
                val_loss = val_metrics['mean_mse']
                history['val_loss'].append(val_loss)

                # Early stopping
                if val_loss < best_val_loss - self.config.min_delta:
                    best_val_loss = val_loss
                    best_model_state = deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.config.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                best_model_state = deepcopy(model.state_dict())

        # Restore best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        return model, history

    def leave_one_out_cv(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform Leave-One-Out cross-validation.

        Returns:
            Dictionary with per-fold results and overall metrics.
        """
        # Prepare data
        df, self.feature_dicts = self.prepare_data(dataframe)

        # Get feature dimensions for model creation
        sample_smiles = [df.cano_smiles.values[0]]
        x_atom, x_bonds, _, _, _, _ = get_smiles_array(sample_smiles, self.feature_dicts)
        num_atom_features = x_atom.shape[-1]
        num_bond_features = x_bonds.shape[-1]

        # Initialize
        loo = LeaveOneOut()
        initial_model = self.create_model(num_atom_features, num_bond_features)
        initial_state = deepcopy(initial_model.state_dict())

        fold_results = []
        host_predictions = {}
        test_predictions = {t: [] for t in self.config.targets}
        test_actuals = {t: [] for t in self.config.targets}

        for fold, (train_idx, test_idx) in enumerate(loo.split(df)):
            print(f"\n--- Fold {fold + 1}/{len(df)} ---")

            # Create fold datasets
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            current_host = test_df[self.config.host_column].values[0] if self.config.host_column in test_df.columns else test_df.cano_smiles.values[0]

            # Split train into train/val
            val_size = max(5, int(0.1 * len(train_df)))
            train_df, val_df = train_test_split(train_df, test_size=val_size)

            # Reset model
            model = self.create_model(num_atom_features, num_bond_features)
            model.load_state_dict(initial_state)

            # Train
            model, history = self.fit(train_df, val_df, model)

            # Evaluate
            train_metrics = self.evaluate(model, train_df)
            val_metrics = self.evaluate(model, val_df)
            test_metrics = self.evaluate(model, test_df)

            # Store fold results
            fold_results.append({
                'fold': fold + 1,
                'host': current_host,
                'train_mse': train_metrics['mean_mse'],
                'val_mse': val_metrics['mean_mse'],
                'test_mse': test_metrics['mean_mse'],
                'train_mae': train_metrics['mean_mae'],
                'test_mae': test_metrics['mean_mae']
            })

            # Store predictions
            for target in self.config.targets:
                test_predictions[target].extend(test_metrics['predictions'][target])
                test_actuals[target].extend(test_metrics['actuals'][target])

            # Store host-specific predictions
            host_predictions[current_host] = {
                target: {
                    'actual': float(test_metrics['actuals'][target][0]),
                    'predicted': float(test_metrics['predictions'][target][0])
                }
                for target in self.config.targets
            }

            print(f"Train MSE: {train_metrics['mean_mse']:.4f}")
            print(f"Val MSE: {val_metrics['mean_mse']:.4f}")
            print(f"Test MSE: {test_metrics['mean_mse']:.4f}")

        # Overall metrics
        overall_mse = np.mean([f['test_mse'] for f in fold_results])
        overall_mae = np.mean([f['test_mae'] for f in fold_results])
        mse_std = np.std([f['test_mse'] for f in fold_results])
        mae_std = np.std([f['test_mae'] for f in fold_results])

        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Test MSE: {overall_mse:.4f} ± {mse_std:.4f}")
        print(f"Test MAE: {overall_mae:.4f} ± {mae_std:.4f}")

        return {
            'fold_results': fold_results,
            'overall_mse': overall_mse,
            'overall_mae': overall_mae,
            'mse_std': mse_std,
            'mae_std': mae_std,
            'host_predictions': host_predictions,
            'test_predictions': test_predictions,
            'test_actuals': test_actuals
        }


# =============================================================================
# 4. UTILITIES
# =============================================================================
def plot_predictions(results: Dict, targets: List[str], save_path: str = None):
    """Plot actual vs predicted values for each target."""
    n_targets = len(targets)
    n_cols = min(4, n_targets)
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_targets == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, target in enumerate(targets):
        ax = axes[i]
        actuals = results['test_actuals'][target]
        predictions = results['test_predictions'][target]

        ax.scatter(actuals, predictions, alpha=0.6, edgecolors='k', linewidth=0.5)

        # Diagonal line
        min_val = min(min(actuals), min(predictions))
        max_val = max(max(actuals), max(predictions))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)

        r2 = r2_score(actuals, predictions)
        ax.set_title(f'{target}\nR² = {r2:.3f}')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')

    for i in range(n_targets, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def print_host_predictions(host_predictions: Dict, targets: List[str]):
    """Print formatted host-specific predictions."""
    print("\n" + "=" * 80)
    print("HOST-SPECIFIC PREDICTIONS")
    print("=" * 80)

    for host, preds in host_predictions.items():
        print(f"\nHost: {host}")
        print("-" * 50)
        for target in targets:
            if target in preds:
                actual = preds[target]['actual']
                predicted = preds[target]['predicted']
                error = abs(actual - predicted)
                print(f"  {target:>10}: Actual={actual:>8.4f}, Pred={predicted:>8.4f}, Err={error:>7.4f}")


# =============================================================================
# 5. MAIN FUNCTION
# =============================================================================
def main():
    """Main entry point for training AttentiveFP model."""
    parser = argparse.ArgumentParser(description='Train AttentiveFP for molecular property prediction')

    # Data arguments
    parser.add_argument('--data_path', type=str, default='../Database/calix smiles small set.csv',
                        help='Path to CSV file with SMILES and targets')
    parser.add_argument('--smiles_column', type=str, default='SMILES',
                        help='Name of SMILES column')
    parser.add_argument('--host_column', type=str, default='Host',
                        help='Name of host/molecule identifier column')
    parser.add_argument('--targets', nargs='+', default=None,
                        help='Target column names')

    # Model arguments
    parser.add_argument('--fingerprint_dim', type=int, default=150)
    parser.add_argument('--radius', type=int, default=3)
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=38)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--patience', type=int, default=30)

    # Output arguments
    parser.add_argument('--plot', action='store_true', help='Generate prediction plots')

    args = parser.parse_args()

    # Create configuration
    config = Config(
        data_path=args.data_path,
        smiles_column=args.smiles_column,
        host_column=args.host_column,
        fingerprint_dim=args.fingerprint_dim,
        radius=args.radius,
        T=args.T,
        p_dropout=args.dropout,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience
    )

    if args.targets:
        config.targets = args.targets

    # Load data
    print(f"Loading data from {config.data_path}")
    print(f"Using device: {config.device}")
    dataframe = pd.read_csv(config.data_path)
    print(f"Loaded {len(dataframe)} molecules")
    print(f"Targets: {config.targets}")

    # Train and evaluate
    trainer = Trainer(config)
    results = trainer.leave_one_out_cv(dataframe)

    # Print results
    print_host_predictions(results['host_predictions'], config.targets)

    # Generate plots
    if args.plot:
        plot_predictions(results, config.targets, save_path='predictions.png')

    return results


# =============================================================================
# 6. ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
