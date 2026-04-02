import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, brentq

# ==========================================
# 1. Assay constants
# ==========================================
L0 = 250e-9       # lucigenin total concentration (M) - 250 nM is constant for all assays

# ==========================================
# 2. Experimental data
# ==========================================
# Direct titration: host added to fixed dye
x_direct_uM = np.array([0, 0.3, 0.7, 0.8, 5.0])
y_direct_RFU = np.array([0, -160, -350, -410, -700])

# Competitive titration: guest added to fixed host + dye
x_comp_uM = np.array([0, 10, 25, 50])
y_comp_RFU = np.array([0, 130, 220, 300])

# EXACT host concentration used in the competition wells (M)
# This MUST be provided by the experimentalist for each specific assay!
H0_comp = 1e-6    

x_direct_M = x_direct_uM * 1e-6
x_comp_M = x_comp_uM * 1e-6

# ==========================================
# 3. Direct titration model
# ==========================================
def direct_signal(H0_array, F0, dF_max, Kd_ind):
    H0_array = np.asarray(H0_array)
    b = H0_array + L0 + Kd_ind
    HL = 0.5 * (b - np.sqrt(b**2 - 4 * H0_array * L0))
    frac_bound = HL / L0
    return F0 + dF_max * frac_bound

p0_direct = [0.0, -800.0, 0.5e-6]
bounds_direct = ([-np.inf, -np.inf, 1e-12], [np.inf, np.inf, 1e-3])

popt_dir, pcov_dir = curve_fit(
    direct_signal,
    x_direct_M,
    y_direct_RFU,
    p0=p0_direct,
    bounds=bounds_direct
)

F0_dir, dF_max_dir, Kd_ind = popt_dir
Kd_ind_err = np.sqrt(np.diag(pcov_dir))[2]

# ==========================================
# 4. Wang-style competitive model
# ==========================================
def solve_free_host(H0, L0, G0, Kd_ind, Kd_guest):
    def mass_balance(H):
        HL = H * L0 / (Kd_ind + H)
        HG = H * G0 / (Kd_guest + H)
        return H + HL + HG - H0

    return brentq(mass_balance, 1e-18, H0)

def competitive_signal_raw(G0_array, Kd_guest):
    G0_array = np.asarray(G0_array)
    y = np.zeros_like(G0_array, dtype=float)

    for i, G0 in enumerate(G0_array):
        H_free = solve_free_host(H0_comp, L0, G0, Kd_ind, Kd_guest)
        HL = H_free * L0 / (Kd_ind + H_free)
        frac_bound_dye = HL / L0
        y[i] = F0_dir + dF_max_dir * frac_bound_dye

    return y

# ==========================================
# 5. Strict competitive model for baseline-subtracted dFobs
# ==========================================
def competitive_signal_delta(G0_array, Kd_guest):
    # Get the raw signal for the provided array
    raw = competitive_signal_raw(G0_array, Kd_guest)
    
    # Explicitly calculate the baseline at G0 = 0 to avoid array-index reliance
    baseline = competitive_signal_raw([0.0], Kd_guest)[0]
    
    # Return dFobs relative to the host-dye complex (G0 = 0)
    return raw - baseline

# ==========================================
# 6. Fit competition data
# ==========================================
p0_comp = [10e-6]
bounds_comp = ([1e-12], [1e-2])

popt_comp, pcov_comp = curve_fit(
    competitive_signal_delta,
    x_comp_M,
    y_comp_RFU,
    p0=p0_comp,
    bounds=bounds_comp
)

Kd_guest = popt_comp[0]
Kd_guest_err = np.sqrt(np.diag(pcov_comp))[0]

# ==========================================
# 7. Plot & Save
# ==========================================

# --- Plot 1: Direct Titration ---
fig1, ax1 = plt.subplots(figsize=(6, 4))
x_smooth_dir = np.linspace(0, x_direct_M.max() * 1.1, 300)

ax1.plot(x_direct_uM, y_direct_RFU, 'o', mfc='white', mec='blue', mew=2)
ax1.plot(x_smooth_dir * 1e6, direct_signal(x_smooth_dir, *popt_dir), 'b-', lw=2)
ax1.set_title('Direct Titration')
ax1.set_xlabel('Host concentration (μM)')
ax1.set_ylabel('dF_obs (RFU)')
ax1.text(
    0.45 * x_direct_uM.max(),
    0.85 * y_direct_RFU.min(),
    f"Kd = {Kd_ind*1e6:.2f} ± {Kd_ind_err*1e6:.2f} μM",
    color='blue',
    fontsize=12
)

plt.tight_layout()
fig1.savefig('./direct_titration.png', dpi=300)
plt.close(fig1) # Close to free up memory

# --- Plot 2: Competition Titration ---
fig2, ax2 = plt.subplots(figsize=(6, 4))
x_smooth_comp = np.linspace(0, x_comp_M.max() * 1.1, 300)

ax2.plot(x_comp_uM, y_comp_RFU, 'o', mfc='white', mec='blue', mew=2)
ax2.plot(
    x_smooth_comp * 1e6,
    competitive_signal_delta(x_smooth_comp, *popt_comp),
    'b-',
    lw=2
)
ax2.set_title('Competition')
ax2.set_xlabel('Guest concentration (μM)')
ax2.set_ylabel('Signal (RFU)')
ax2.text(
    0.35 * x_comp_uM.max(),
    0.85 * y_comp_RFU.max(),
    f"Kd = {Kd_guest*1e6:.2f} ± {Kd_guest_err*1e6:.2f} μM",
    color='blue',
    fontsize=12
)

plt.tight_layout()
fig2.savefig('./competitive_titration.png', dpi=300)
plt.close(fig2)

# ==========================================
# 8. Terminal Output
# ==========================================
print(f"Direct Kd (indicator): {Kd_ind*1e6:.4f} ± {Kd_ind_err*1e6:.4f} μM")
print(f"Competition Kd (guest): {Kd_guest*1e6:.4f} ± {Kd_guest_err*1e6:.4f} μM")
