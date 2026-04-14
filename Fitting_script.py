import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, brentq

# ==========================================
# 1. Load Data and Fix Outliers
# ==========================================
df = pd.read_excel('device_reading.xlsx')
df.columns = df.columns.astype(str)

x_direct_uM = pd.to_numeric(df.iloc[2:8, 0]).values 
y_direct_1 = pd.to_numeric(df.iloc[2:8, 2]).values
y_direct_2 = pd.to_numeric(df.iloc[2:8, 3]).values

df.iloc[5, df.columns.get_loc('10')] = df.iloc[4, df.columns.get_loc('10')] 

L0 = 250e-9       # Indicator concentration (250 nM)
H0_comp = 0.5e-6  # Host concentration for competitive assay (0.5 µM)

peptides = [
    ("H3K4",      0, 2, ['0.5', '1.5', '10', '25', '50']),
    ("H3K4me3",   0, 2, ['0.5.1', '1.5.1', '10.1', '25.1', '50.1']),
    ("H3K4Ac",    2, 4, ['0.5', '1.5', '10', '25', '50']),
    ("H3K9me3",   2, 4, ['0.5.1', '1.5.1', '10.1', '25.1', '50.1']),
    ("H3K4me",    4, 6, ['0.5', '1.5', '10', '25', '50']),
    ("H3R2me2a",  4, 6, ['0.5.1', '1.5.1', '10.1', '25.1', '50.1']),
    ("H3K4me2",   6, 8, ['0.5', '1.5', '10', '25', '50']),
    ("H3R2me2s",  6, 8, ['0.5.1', '1.5.1', '10.1', '25.1', '50.1']),
]

# Read concentrations from the Excel file instead of hardcoding
x_comp_uM = np.array([float(c) for c in ['0.5', '1.5', '10', '25', '50']])  # from column headers
x_comp_M = x_comp_uM * 1e-6
x_direct_M = x_direct_uM * 1e-6

# ==========================================
# 3. Fit Direct Titrations (Separate)
# ==========================================
def direct_signal(H0_array, F0, dF_max, Kd_ind):
    H0_array = np.asarray(H0_array)
    b = H0_array + L0 + Kd_ind
    HL = 0.5 * (b - np.sqrt(b**2 - 4 * H0_array * L0))
    frac_bound = HL / L0
    return F0 + dF_max * frac_bound

p0_direct = [0.0, -800.0, 0.5e-6]
bounds_direct = ([-np.inf, -np.inf, 1e-12], [np.inf, np.inf, 1e-3])

popt_dir1, pcov_dir1 = curve_fit(direct_signal, x_direct_M, y_direct_1, p0=p0_direct, bounds=bounds_direct)
Kd_ind1 = popt_dir1[2]
Kd_ind_err1 = np.sqrt(np.diag(pcov_dir1))[2]

popt_dir2, pcov_dir2 = curve_fit(direct_signal, x_direct_M, y_direct_2, p0=p0_direct, bounds=bounds_direct)
Kd_ind2 = popt_dir2[2]
Kd_ind_err2 = np.sqrt(np.diag(pcov_dir2))[2]

# ==========================================
# 4. Fit Competitive Titrations (Scaled Model)
# ==========================================
def create_scaled_competitive_model(Kd_ind):
    def solve_free_host(H0, L0, G0, Kd_guest):
        def mass_balance(H):
            HL = H * L0 / (Kd_ind + H)
            HG = H * G0 / (Kd_guest + H)
            return H + HL + HG - H0
        return brentq(mass_balance, 1e-18, H0)

    def competitive_signal_raw(G0_array, dF_comp_max, Kd_guest):
        G0_array = np.asarray(G0_array)
        y = np.zeros_like(G0_array, dtype=float)
        for i, G0 in enumerate(G0_array):
            H_free = solve_free_host(H0_comp, L0, G0, Kd_guest)
            HL = H_free * L0 / (Kd_ind + H_free)
            frac_bound_dye = HL / L0
            # Flip the sign for competitive displacement
            y[i] = -dF_comp_max * frac_bound_dye
        return y

    def competitive_signal_delta(G0_array, dF_comp_max, Kd_guest):
        raw = competitive_signal_raw(G0_array, dF_comp_max, Kd_guest)
        baseline = competitive_signal_raw([0.0], dF_comp_max, Kd_guest)[0]
        return raw - baseline
    return competitive_signal_delta

comp_model_1 = create_scaled_competitive_model(Kd_ind1)
comp_model_2 = create_scaled_competitive_model(Kd_ind2)

def calc_sey(y_obs, y_pred, p=1):
    n = len(y_obs)
    if n > p:
        return np.sqrt(np.sum((y_obs - y_pred)**2) / (n - p))
    return np.nan

y_fit_dir1 = direct_signal(x_direct_M, *popt_dir1)
sey_dir1 = calc_sey(y_direct_1, y_fit_dir1)
y_fit_dir2 = direct_signal(x_direct_M, *popt_dir2)
sey_dir2 = calc_sey(y_direct_2, y_fit_dir2)

# ==========================================
# 5. Plotting
# ==========================================
import os
os.makedirs('output', exist_ok=True)

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 12

fig, axes_2d = plt.subplots(3, 3, figsize=(10, 9.5))
fig.subplots_adjust(wspace=0.1, hspace=0.45)
axes = axes_2d.flatten()

# --- Panel 0: Direct Titration ---
ax = axes[0]
x_smooth_dir = np.linspace(0, x_direct_M.max() * 1.1, 300)

ax.plot(x_smooth_dir * 1e6, direct_signal(x_smooth_dir, *popt_dir1), color='#d62728', lw=2)
ax.plot(x_smooth_dir * 1e6, direct_signal(x_smooth_dir, *popt_dir2), color='#10239e', lw=2)

ax.plot(x_direct_uM, y_direct_1, 'o', mfc='none', mec='#d62728', mew=1.5)
ax.plot(x_direct_uM, y_direct_2, 'o', mfc='none', mec='#10239e', mew=1.5)

text_red_dir = f"K$_d$ = {Kd_ind1*1e6:.2f} ± {Kd_ind_err1*1e6:.2f} µM\nSE$_y$={sey_dir1:.1f}"
text_blue_dir = f"K$_d$ = {Kd_ind2*1e6:.2f} ± {Kd_ind_err2*1e6:.2f} µM\nSE$_y$={sey_dir2:.1f}"
ax.text(0.35, 0.93, text_red_dir, transform=ax.transAxes, color='#d62728', fontsize=11, va='top', linespacing=1.5)
ax.text(0.35, 0.73, text_blue_dir, transform=ax.transAxes, color='#10239e', fontsize=11, va='top', linespacing=1.5)

ax.set_title('Direct Titration', fontsize=14, pad=10)
ax.set_xlim(-0.2, 5.2)
ax.set_xticks([0, 1.0, 2.5, 5.0])
ax.set_xlabel('Host concentration (µM)', fontsize=13)
ax.set_ylim(-820, 10)
ax.set_yticks([0, -200, -400, -600, -800])

# --- Panels 1-8: Competitive ---
for i, (pep_name, row_start, row_end, cols) in enumerate(peptides):
    idx = i + 1
    ax = axes[idx]

    y_rep1 = pd.to_numeric(df.iloc[row_start, [df.columns.get_loc(c) for c in cols]]).values
    y_rep2 = pd.to_numeric(df.iloc[row_start+1, [df.columns.get_loc(c) for c in cols]]).values

    # We now fit [dF_comp_max, Kd_guest]
    p0_comp = [800.0, 10e-6]
    bounds_comp = ([0.0, 1e-12], [np.inf, 1e-2])
    x_smooth_comp = np.linspace(0, x_comp_M.max() * 1.1, 300)

    try:
        popt1, pcov1 = curve_fit(comp_model_1, x_comp_M, y_rep1, p0=p0_comp, bounds=bounds_comp)
        Kd_g1, err1 = popt1[1], np.sqrt(np.diag(pcov1))[1]
        sey1 = calc_sey(y_rep1, comp_model_1(x_comp_M, *popt1))
        ax.plot(x_smooth_comp * 1e6, comp_model_1(x_smooth_comp, *popt1), color='#d62728', lw=2)
    except Exception:
        Kd_g1, err1, sey1 = np.nan, np.nan, np.nan

    try:
        popt2, pcov2 = curve_fit(comp_model_2, x_comp_M, y_rep2, p0=p0_comp, bounds=bounds_comp)
        Kd_g2, err2 = popt2[1], np.sqrt(np.diag(pcov2))[1]
        sey2 = calc_sey(y_rep2, comp_model_2(x_comp_M, *popt2))
        ax.plot(x_smooth_comp * 1e6, comp_model_2(x_smooth_comp, *popt2), color='#10239e', lw=2)
    except Exception:
        Kd_g2, err2, sey2 = np.nan, np.nan, np.nan

    ax.plot(x_comp_uM, y_rep1, 'o', mfc='none', mec='#d62728', mew=1.5)
    ax.plot(x_comp_uM, y_rep2, 'o', mfc='none', mec='#10239e', mew=1.5)

    text_red = f"K$_d$ ={Kd_g1*1e6:.2f} ± {err1*1e6:.2f} µM  SE$_y$={sey1:.1f}"
    text_blue = f"K$_d$ ={Kd_g2*1e6:.2f} ± {err2*1e6:.2f} µM  SE$_y$={sey2:.1f}"
    ax.text(0.04, 0.93, text_red, transform=ax.transAxes, color='#d62728', fontsize=10, va='top')
    ax.text(0.04, 0.83, text_blue, transform=ax.transAxes, color='#10239e', fontsize=10, va='top')

    ax.set_title(pep_name, fontsize=14, pad=10)
    ax.set_xlim(-2, 55)
    ax.set_ylim(-10, 820)
    ax.set_yticks([0, 200, 400, 600, 800])

    # Bottom row only: X labels
    if idx >= 6:
        ax.set_xticks([0, 10, 25, 50])
        ax.set_xlabel('Guest concentration (µM)', fontsize=13)
    else:
        ax.set_xticks([])

# Global Y-axis Styling
for idx, ax in enumerate(axes):
    row, col = divmod(idx, 3)

    if col == 0:
        ax.set_ylabel(r'dF$_{obs}$ (RFU)', fontsize=13)
    elif col == 1:
        ax.set_yticklabels([])
    elif col == 2:
        ax.yaxis.tick_right()

plt.subplots_adjust(right=0.92)
plt.savefig('output/titration_9panels.png', dpi=300, bbox_inches='tight')
plt.savefig('output/titration_9panels.pdf', bbox_inches='tight')
print("9-panel plot generated (PNG + PDF).")
