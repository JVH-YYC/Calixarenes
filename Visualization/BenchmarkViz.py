"""
Null-model benchmark visualization for calixarene paper revisions.

Generates predicted-vs-actual scatter plots for three null baselines:
    (i)   Uniform random prediction (bounded by dataset min/max)
    (ii)  Global mean prediction
    (iii) Per-guest (peptide) mean prediction

Plots are produced for both the full dataset and the small (predictable) subset.
R2 is reported on each plot to match the metric used throughout the paper.

Styling is matched to the existing CalixViz / CreateCalixFigure conventions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Plot settings (matched to CreateCalixFigure.py / CalixViz.py conventions)
# ---------------------------------------------------------------------------
PLOT_SETTINGS = {
    'fig_width': 10,
    'fig_height': 10,
    'axis_font_size': 16,
    'tick_font_size': 12,
    'title_font_size': 20,
    'font_type': 'DejaVu Sans',
    'scatter_size': 25,
    'scatter_opacity': 0.5,
    'scatter_color': 'orange',
    'scatter_marker': 'o',
    'dpi': 300,
}

# Peptide (guest) columns in the CSV files
PEPTIDE_COLUMNS = ['H3K4', 'H3K4ac', 'H3K4me1', 'H3K4me2',
                   'H3K4me3', 'H3K9me3', 'H3R2me2a', 'H3R2me2s']

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(csv_path):
    """
    Load a calixarene CSV and return a long-form DataFrame with columns:
        Host, Peptide, Actual
    """
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        host = row['Host']
        for pep in PEPTIDE_COLUMNS:
            records.append({'Host': host, 'Peptide': pep, 'Actual': row[pep]})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Null-model predictions
# ---------------------------------------------------------------------------

def predict_uniform_random(actual_values, seed=42):
    """Return random predictions sampled uniformly between dataset min and max."""
    rng = np.random.default_rng(seed)
    lo, hi = actual_values.min(), actual_values.max()
    return rng.uniform(lo, hi, size=len(actual_values))


def predict_global_mean(actual_values):
    """Return the global mean repeated for every example."""
    return np.full(len(actual_values), actual_values.mean())


def predict_per_guest_mean(df):
    """
    Return an array of predictions where each example is predicted as the
    mean actual value for its peptide (guest).
    """
    guest_means = df.groupby('Peptide')['Actual'].transform('mean')
    return guest_means.values


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_square_scatter(actual, predicted, title, output_path,
                         ps=PLOT_SETTINGS, save_fig=True):
    """
    Produce a single predicted-vs-actual scatter plot with R2 annotation,
    diagonal reference line, and square axes.  Matches CalixViz styling.
    """
    r2 = r2_score(actual, predicted)

    fig, ax = plt.subplots(figsize=(ps['fig_width'], ps['fig_height']))

    ax.scatter(predicted, actual,
               color=ps['scatter_color'],
               s=ps['scatter_size'],
               alpha=ps['scatter_opacity'],
               marker=ps['scatter_marker'])

    # Square axes with matching limits
    all_vals = np.concatenate([actual, predicted])
    vmin, vmax = all_vals.min(), all_vals.max()
    pad = (vmax - vmin) * 0.05
    vmin -= pad
    vmax += pad
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)

    # Diagonal reference line
    ax.plot([vmin, vmax], [vmin, vmax],
            color='black', linestyle='--', linewidth=1)

    # Labels and title
    ax.set_xlabel('Predicted ln(Kd)', fontsize=ps['axis_font_size'],
                  fontname=ps['font_type'])
    ax.set_ylabel('Actual ln(Kd)', fontsize=ps['axis_font_size'],
                  fontname=ps['font_type'])
    ax.set_title(title, fontsize=ps['title_font_size'],
                 fontname=ps['font_type'])
    ax.tick_params(labelsize=ps['tick_font_size'])

    # R2 annotation in upper-left
    ax.text(0.05, 0.95, f'$R^2 = {r2:.3f}$',
            transform=ax.transAxes,
            fontsize=ps['axis_font_size'],
            fontname=ps['font_type'],
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='black', alpha=0.8))

    if save_fig:
        fig.savefig(output_path, dpi=ps['dpi'], facecolor='white',
                    bbox_inches='tight', pad_inches=0.05, format='png')
        print(f'Saved: {output_path}')

    plt.show()
    plt.close(fig)
    return r2


def generate_all_benchmarks(csv_path, dataset_label, output_folder,
                            seed=42, save_fig=True):
    """
    For a single dataset CSV, produce the three null-model plots and print R2.

    Parameters
    ----------
    csv_path : str
        Path to the calixarene CSV (full or small set).
    dataset_label : str
        Human-readable label, e.g. "Full Set" or "Small Set".
        Used in plot titles and output file names.
    output_folder : str
        Directory in which to save PNG files.
    seed : int
        Random seed for the uniform-random null model.
    save_fig : bool
        Whether to write PNG files to disk.
    """
    os.makedirs(output_folder, exist_ok=True)
    df = load_dataset(csv_path)
    actual = df['Actual'].values
    safe_label = dataset_label.replace(' ', '_')

    results = {}

    # (i) Uniform random
    pred_random = predict_uniform_random(actual, seed=seed)
    r2_rand = _make_square_scatter(
        actual, pred_random,
        title=f'Null Model: Uniform Random ({dataset_label})',
        output_path=os.path.join(output_folder,
                                 f'Null_Uniform_Random_{safe_label}.png'),
        save_fig=save_fig)
    results['Uniform Random'] = r2_rand

    # (ii) Global mean
    pred_mean = predict_global_mean(actual)
    r2_mean = _make_square_scatter(
        actual, pred_mean,
        title=f'Null Model: Global Mean ({dataset_label})',
        output_path=os.path.join(output_folder,
                                 f'Null_Global_Mean_{safe_label}.png'),
        save_fig=save_fig)
    results['Global Mean'] = r2_mean

    # (iii) Per-guest (peptide) mean
    pred_guest = predict_per_guest_mean(df)
    r2_guest = _make_square_scatter(
        actual, pred_guest,
        title=f'Null Model: Per-Guest Mean ({dataset_label})',
        output_path=os.path.join(output_folder,
                                 f'Null_Per_Guest_Mean_{safe_label}.png'),
        save_fig=save_fig)
    results['Per-Guest Mean'] = r2_guest

    # Summary
    print(f'\n--- Null-model R2 summary ({dataset_label}) ---')
    for method, r2_val in results.items():
        print(f'  {method:20s}  R2 = {r2_val:.4f}')

    return results


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Paths relative to project root (run from Calixarenes/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    feat_folder = os.path.join(project_root, 'Featurization')
    output_folder = os.path.join(project_root, 'Visualization', 'BenchmarkPlots')

    print('=== Full Dataset ===')
    generate_all_benchmarks(
        csv_path=os.path.join(feat_folder, 'calix smiles absolute.csv'),
        dataset_label='Full Set',
        output_folder=output_folder)

    print('\n=== Small (Predictable) Dataset ===')
    generate_all_benchmarks(
        csv_path=os.path.join(feat_folder, 'calix smiles small set.csv'),
        dataset_label='Small Set',
        output_folder=output_folder)
