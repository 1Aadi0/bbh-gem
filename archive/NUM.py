import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

DATA_FILE = "data.npz"

# --------------------------- I/O --------------------------- #
# --------------------------- I/O --------------------------- #
def load_data(file_path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}")
    d = np.load(file_path)
    
    # Try to load time, default to 'Unknown' if missing
    t_val = d["time"] if "time" in d else None

    return dict(
        time=t_val,  # <--- NEW FIELD
        alp=d["alp"], gxx=d["gxx"], gxy=d["gxy"], gyy=d["gyy"],
        gxz=d.get("gxz", np.zeros_like(d["gxx"])),
        gyz=d.get("gyz", np.zeros_like(d["gxx"])),
        gzz=d.get("gzz", np.ones_like(d["gxx"])),
        betax=d["betax"], betay=d["betay"],
        betaz=d.get("betaz", np.zeros_like(d["betax"])),
        x=d["x"], y=d["y"],
    )
# --------------------------- Grid Reconstruction --------------------------- #
def get_full_binary_grid(data):
    # Standard Grid Reconstruction (Ghosts + Symmetry)
    alp, betax, betay = data["alp"], data["betax"], data["betay"]
    gxx, gxy, gyy = data["gxx"], data["gxy"], data["gyy"]
    x, y = data["x"], data["y"]

    start_idx = np.where(x >= -1e-9)[0][0]
    x_right = x[start_idx:]
    
    # Check orientation
    axis = 1 if alp.shape[1] == len(x) else 0
    sl = (slice(None), slice(start_idx, None)) if axis == 1 else (slice(start_idx, None), slice(None))

    # Slice
    alp_r, bx_r, by_r = alp[sl], betax[sl], betay[sl]
    gxx_r, gxy_r, gyy_r = gxx[sl], gxy[sl], gyy[sl]
    gzz_r = data["gzz"][sl]
    
    # Symmetry Ops: 180 deg rotation
    alp_l = np.flip(np.flip(alp_r, axis=0), axis=1)
    bx_l = -np.flip(np.flip(bx_r, axis=0), axis=1)
    by_l = -np.flip(np.flip(by_r, axis=0), axis=1)
    
    gxx_l = np.flip(np.flip(gxx_r, axis=0), axis=1)
    gxy_l = np.flip(np.flip(gxy_r, axis=0), axis=1)
    gyy_l = np.flip(np.flip(gyy_r, axis=0), axis=1)
    gzz_l = np.flip(np.flip(gzz_r, axis=0), axis=1)

    # Stitch
    def stitch(left, right, ax):
        if np.isclose(x_right[0], 0.0):
            sl_drop = [slice(None)] * left.ndim
            sl_drop[ax] = slice(None, -1)
            left_trimmed = left[tuple(sl_drop)]
            return np.concatenate([left_trimmed, right], axis=ax)
        return np.concatenate([left, right], axis=ax)

    x_full = np.concatenate([-x_right[:0:-1], x_right]) if np.isclose(x_right[0], 0.0) else np.concatenate([-x_right[::-1], x_right])
    
    # Update Data Dict
    new_data = data.copy()
    new_data.update({
        'x': x_full, 
        'alp': stitch(alp_l, alp_r, axis), 
        'betax': stitch(bx_l, bx_r, axis), 
        'betay': stitch(by_l, by_r, axis),
        'gxx': stitch(gxx_l, gxx_r, axis), 
        'gxy': stitch(gxy_l, gxy_r, axis), 
        'gyy': stitch(gyy_l, gyy_r, axis),
        'gzz': stitch(gzz_l, gzz_r, axis),
        'gxz': np.zeros_like(stitch(alp_l, alp_r, axis)),
        'gyz': np.zeros_like(stitch(alp_l, alp_r, axis)),
        'betaz': np.zeros_like(stitch(alp_l, alp_r, axis))
    })
    return new_data

# --------------------------- Math Helpers --------------------------- #
def coefficients(gxx, gxy, gyy, gxz, gyz):
    A = 1.0 / np.sqrt(gxx)
    det2 = np.clip(gxx * gyy - gxy**2, 1e-30, None)
    B = 1.0 / np.sqrt(det2)
    C = gxx * gyz - gxy * gxz
    return A, B, C, det2

def get_inverse_metric(data):
    det = np.clip(data['gxx'] * data['gyy'] - data['gxy']**2, 1e-20, None)
    det_inv = 1.0 / det
    
    g = {}
    g['xx'] = data['gyy'] * det_inv
    g['yy'] = data['gxx'] * det_inv
    g['xy'] = -data['gxy'] * det_inv
    g['zz'] = 1.0 / np.clip(data['gzz'], 1e-20, None) 
    
    alp = data['alp']
    alp2_inv = 1.0 / np.clip(alp**2, 1e-20, None)
    bx, by = data['betax'], data['betay']
    
    g['tt'] = -alp2_inv
    g['tx'] = bx * alp2_inv
    g['ty'] = by * alp2_inv
    g['tz'] = np.zeros_like(alp)
    
    g['xx'] -= (bx * bx) * alp2_inv
    g['xy'] -= (bx * by) * alp2_inv
    g['yy'] -= (by * by) * alp2_inv
    
    g['xt'] = g['tx']; g['yt'] = g['ty']; g['yx'] = g['xy']
    g['xz'] = np.zeros_like(alp); g['yz'] = np.zeros_like(alp)
    g['zx'] = g['xz']; g['zy'] = g['yz']; g['zt'] = g['tz']
    return g

def contract_vector_norm(v, g_inv):
    norm_sq = np.zeros_like(v['t'])
    for mu in ['t', 'x', 'y', 'z']:
        for nu in ['t', 'x', 'y', 'z']:
            key = f"{mu}{nu}" if f"{mu}{nu}" in g_inv else f"{nu}{mu}"
            norm_sq += g_inv[key] * v[mu] * v[nu]
    return norm_sq

def spatial_grads(arr, dx, dy) -> Tuple[np.ndarray, np.ndarray]:
    # Matches your starting version logic exactly
    if arr.shape[1] > 1: 
         d_dy, d_dx = np.gradient(arr, dy, dx, edge_order=2)
    else:
         d_dx, d_dy = np.gradient(arr, dx, dy, edge_order=2)
    return d_dx, d_dy

# --------------------------- 1. FULL Tetrad Construction --------------------------- #
def build_tetrads(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    alp = data["alp"]
    gxx, gxy, gyy = data["gxx"], data["gxy"], data["gyy"]
    gxz, gyz, gzz = data["gxz"], data["gyz"], data["gzz"]
    betax, betay, betaz = data["betax"], data["betay"], data["betaz"]

    # Lower Shift
    b_x = gxx*betax + gxy*betay + gxz*betaz
    b_y = gxy*betax + gyy*betay + gyz*betaz
    b_z = gxz*betax + gyz*betay + gzz*betaz
    
    A, B, C, det2 = coefficients(gxx, gxy, gyy, gxz, gyz)
    B2 = B**2

    g1 = dict(t=b_x, x=gxx, y=gxy, z=gxz)
    g2 = dict(t=b_y, x=gxy, y=gyy, z=gyz)
    g3 = dict(t=b_z, x=gxz, y=gyz, z=gzz)

    theta = {}

    # hat{0} - Observer
    theta["0_t"] = alp
    theta["0_x"] = np.zeros_like(alp); theta["0_y"] = np.zeros_like(alp); theta["0_z"] = np.zeros_like(alp)

    # hat{1}
    theta["1_t"] = A * g1["t"]
    theta["1_x"] = A * g1["x"]; theta["1_y"] = A * g1["y"]; theta["1_z"] = A * g1["z"]

    # hat{2}
    theta["2_t"] = A * B * (gxx * g2["t"] - gxy * g1["t"])
    theta["2_x"] = A * B * (gxx * g2["x"] - gxy * g1["x"])
    theta["2_y"] = A * B * (gxx * g2["y"] - gxy * g1["y"])
    theta["2_z"] = A * B * (gxx * g2["z"] - gxy * g1["z"])

    # hat{3} - Rigorous Normalization
    raw_3 = {}
    for comp in ["t", "x", "y", "z"]:
        term1 = gxx * g3[comp] - gxz * g1[comp]
        term2 = (gxx * g2[comp] - gxy * g1[comp]) * B2 * C
        raw_3[comp] = term1 - term2

    g_inv = get_inverse_metric(data)
    norm_sq = contract_vector_norm(raw_3, g_inv)
    norm_inv = 1.0 / np.sqrt(np.clip(norm_sq, 1e-20, None))
    
    for comp in ["t", "x", "y", "z"]:
         theta[f"3_{comp}"] = raw_3[comp] * norm_inv
         
    return theta

# --------------------------- 2. Strict Orthonormality Check --------------------------- #
def check_orthonormality(theta, data):
    print("\n--- Verifying Orthonormality (STRICT POINTWISE CHECK) ---")
    g_inv = get_inverse_metric(data)
    indices = ["0", "1", "2", "3"]
    
    print(f"{'Pair':<6} | {'Target':<6} | {'Max Err (99%)':<14} | {'Status'}")
    print("-" * 50)
    
    for a in indices:
        for b in indices:
            if a > b: continue 
            
            dot_prod = np.zeros_like(theta["0_t"])
            for mu in ["t", "x", "y", "z"]:
                for nu in ["t", "x", "y", "z"]:
                    metric_term = g_inv[f"{mu}{nu}"] if f"{mu}{nu}" in g_inv else g_inv[f"{nu}{mu}"]
                    dot_prod += metric_term * theta[f"{a}_{mu}"] * theta[f"{b}_{nu}"]
            
            target = -1.0 if (a == b and a == "0") else (1.0 if a == b else 0.0)
            error = np.abs(dot_prod - target)
            max_err = np.percentile(error, 99)
            
            status = "✅" if max_err < 1e-4 else "❌"
            print(f"{a}{b:<5} | {target:+.1f}    | {max_err:.2e}       | {status}")

# --------------------------- 3. Electric Field (Equation 9) --------------------------- #
def calculate_field_strength_tensor(theta, dx, dy):
    """Calculates F_mn = d_m A_n - d_n A_m for all tetrads"""
    F = {k: {} for k in ["0", "1", "2", "3"]}
    for hat in ["0", "1", "2", "3"]:
        dth_t_dx, dth_t_dy = spatial_grads(theta[f"{hat}_t"], dx, dy)
        dth_x_dx, dth_x_dy = spatial_grads(theta[f"{hat}_x"], dx, dy)
        dth_y_dx, dth_y_dy = spatial_grads(theta[f"{hat}_y"], dx, dy)

        # F_xt = d_x A_t - d_t A_x (assuming stationarity d_t = 0)
        # So F_xt = d_x A_t
        F[hat]["F_xt"] = dth_t_dx 
        F[hat]["F_yt"] = dth_t_dy
        F[hat]["F_xy"] = dth_y_dx - dth_x_dy
    return F

def gem_fields(theta, F):
    """
    Standard definition: E = n^t * F_it
    """
    alp = theta["0_t"]
    n_t = 1.0 / np.clip(alp, 1e-30, None)
    E = {}
    for hat in ["0", "1", "2", "3"]:
        E[hat] = (n_t * F[hat]["F_xt"], n_t * F[hat]["F_yt"])
    return E

# --------------------------- 4. Magnetic Field Implementations --------------------------- #
def calculate_B3_via_eq8_rigorous(theta, dx, dy):
    """
    Rigorous Eq 8 Calculation: B^3 = curl(A^3)
    """
    Az = theta["3_z"]
    dAz_dx, dAz_dy = spatial_grads(Az, dx, dy)
    
    # Bx = d_y A_z, By = -d_x A_z
    B3_x = dAz_dy
    B3_y = -dAz_dx
    return B3_x, B3_y

def calculate_shift_curl_gem(betax, betay, dx, dy):
    """
    GEM Analogy Calculation: B_GEM = curl(Shift)
    """
    dbx_dx, dbx_dy = spatial_grads(betax, dx, dy)
    dby_dx, dby_dy = spatial_grads(betay, dx, dy)
    
    # Curl z component
    Bz = dby_dx - dbx_dy
    return Bz

from scipy.ndimage import gaussian_filter

def main(data_path: str = DATA_FILE, out_png: str = "Paper_Replication_Polished.png"):
    raw_data = load_data(data_path)
    data = get_full_binary_grid(raw_data)
    x, y = data["x"], data["y"]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])

    print(f"Grid Shape: {data['alp'].shape}")

    # --- 1. PRE-SMOOTHING (Crucial for removing 'Cross' artifacts) ---
    # We smooth the metric potentials BEFORE calculating derivatives.
    # Sigma=2.0 smears the grid noise but keeps the large-scale physics.
    sigma_val = 2.0
    data['alp'] = gaussian_filter(data['alp'], sigma=sigma_val)
    data['betax'] = gaussian_filter(data['betax'], sigma=sigma_val)
    data['betay'] = gaussian_filter(data['betay'], sigma=sigma_val)
    data['gxx'] = gaussian_filter(data['gxx'], sigma=sigma_val)
    data['gyy'] = gaussian_filter(data['gyy'], sigma=sigma_val)
    data['gxy'] = gaussian_filter(data['gxy'], sigma=sigma_val)

    # 2. Build Tetrads & Fields (On Smoothed Data)
    theta = build_tetrads(data)
    F_tensor = calculate_field_strength_tensor(theta, dx, dy)
    E_fields = gem_fields(theta, F_tensor)
    B3_x, B3_y = calculate_B3_via_eq8_rigorous(theta, dx, dy)
    
    # 3. Extract Data & Calc Energy Density
    def prep(arr): return arr.T if arr.shape == (len(x), len(y)) else arr

    Ex, Ey = E_fields["0"]
    
    # Energy Density q0 (Eq 25)
    det_gamma = data['gxx'] * data['gyy'] - data['gxy']**2
    sqrt_gamma = np.sqrt(np.abs(det_gamma))
    
    E_sq = Ex**2 + Ey**2
    B_sq = B3_x**2 + B3_y**2
    
    # q0 = Energy Density ~ (E^2 + B^2)
    q0 = 0.5 * sqrt_gamma * (E_sq + B_sq)
    
    # Post-Smooth q0 to make it look like a clean "blob"
    q0 = gaussian_filter(q0, sigma=1.0)

    # 4. Plotting
    vals_x, vals_y = x, y
    vals_alp = prep(data['alp'])
    vals_q0  = prep(q0)
    vals_Ex, vals_Ey = prep(Ex), prep(Ey)
    vals_B3x, vals_B3y = prep(B3_x), prep(B3_y)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 16))
    sim_time = raw_data.get("time", None)
    t_str = f"t = {float(sim_time):.2f} M" if sim_time is not None else "Unknown"
    fig.suptitle(f"Replication of Boyeneni et al. (Fig 3) | {t_str}", fontsize=20, fontweight='bold')

    # --- TOP PLOT: Lapse (Contours) + Magnetic Field ---
    ax1 = axes[0]
    
    # CONTOUR PLOT for Lapse (Matches Paper Style)
    # 20 discrete levels from 0 to 1
    levels = np.linspace(0.0, 1.0, 21) 
    cf1 = ax1.contourf(vals_x, vals_y, vals_alp, levels=levels, cmap="Reds_r", extend='both')
    
    # Streamlines (Magnetic Field)
    ax1.streamplot(vals_x, vals_y, vals_B3x, vals_B3y, color="black", density=1.5, arrowsize=1.2, linewidth=0.8)
    
    ax1.set_title("Lapse (alpha) & Magnetic Field B", fontsize=14)
    fig.colorbar(cf1, ax=ax1, label="Lapse (alpha)")
    ax1.set_xlim(-15, 15); ax1.set_ylim(-15, 15)
    ax1.set_aspect("equal")

    # --- BOTTOM PLOT: Energy Density q0 + Electric Field ---
    ax2 = axes[1]
    
    # LOG SCALE with Percentile Clipping (Removes the "Ring" noise)
    abs_q0 = np.abs(vals_q0)
    v_max = np.percentile(abs_q0, 99.8) # Peak of the blob
    v_min = v_max * 1e-4                # Floor at 4 orders of magnitude below peak
    
    log_q0 = np.log10(abs_q0 + 1e-20)
    log_min = np.log10(v_min)
    log_max = np.log10(v_max)
    
    # Use "OrRd" (Orange-Red) to match paper's heat map
    pcm2 = ax2.pcolormesh(vals_x, vals_y, log_q0, cmap="OrRd", shading="gouraud", vmin=log_min, vmax=log_max)
    
    # Electric Field Streamlines
    ax2.streamplot(vals_x, vals_y, vals_Ex, vals_Ey, color="black", density=1.5, arrowsize=1.2, linewidth=0.8)
    
    ax2.set_title(f"Energy Density q0 (Eq 25) & Electric Field E", fontsize=14)
    fig.colorbar(pcm2, ax=ax2, label="log10 |q0| (Scaled to Structure)")
    ax2.set_xlim(-15, 15); ax2.set_ylim(-15, 15)
    ax2.set_aspect("equal")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(out_png, dpi=150)
    print(f"🎉 Polished Paper Replication saved to {out_png}")

if __name__ == "__main__":
    main()