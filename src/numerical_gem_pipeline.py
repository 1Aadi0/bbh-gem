import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

DATA_FILE = "final_data.npz"

# --------------------------- I/O --------------------------- #
def load_data(file_path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}")
    d = np.load(file_path)
    return dict(
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

# --------------------------- Main --------------------------- #
def main(data_path: str = DATA_FILE, out_png: str = "Inspiral_Fields.png"):
    raw_data = load_data(data_path)
    data = get_full_binary_grid(raw_data)
    x, y = data["x"], data["y"]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])

    print(f"Grid Shape: {data['alp'].shape}")

    # 1. Build & Verify Full Tetrads
    theta = build_tetrads(data)
    check_orthonormality(theta, data)

    # 2. Calculate Fields (Raw, no filtering)
    F_tensor = calculate_field_strength_tensor(theta, dx, dy)
    
    # Use the specific user-requested definition (Eq 9)
    E_fields = gem_fields(theta, F_tensor)
    
    # 3. Calculate Magnetic Fields (Both versions)
    B3_x, B3_y = calculate_B3_via_eq8_rigorous(theta, dx, dy)
    Bz_GEM = calculate_shift_curl_gem(data["betax"], data["betay"], dx, dy)

    # 4. Plotting Prep
    def prep(arr):
        if arr.shape == (len(y), len(x)): return arr
        return arr.T

    # Extract E^0 components (Observer)
    Ex_raw, Ey_raw = E_fields["0"]

    vals_x, vals_y = x, y
    vals_alp = prep(data['alp'])
    vals_Ex, vals_Ey = prep(Ex_raw), prep(Ey_raw)
    vals_Bz_GEM = prep(Bz_GEM)
    vals_betax, vals_betay = prep(data['betax']), prep(data['betay'])
    vals_B3x, vals_B3y = prep(B3_x), prep(B3_y)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Plot 1: E-Field (Raw Dipole)
    ax = axes[0, 0]
    ax.pcolormesh(vals_x, vals_y, vals_alp, cmap="Reds", shading="auto", vmin=0.0, vmax=1.0)
    ax.streamplot(vals_x, vals_y, vals_Ex, vals_Ey, color="black", density=1.2, arrowsize=1.0)
    ax.set_title("1. Electric Field Lines (Eq 9, No Filter)")
    ax.set_aspect("equal")

    # Plot 2: Magnetic Field (Rigorous Eq 8 - Saddle)
    limit = np.max(np.abs(vals_Bz_GEM)) * 0.8
    ax = axes[0, 1]
    ax.pcolormesh(vals_x, vals_y, vals_Bz_GEM, cmap="RdBu_r", shading="auto", vmin=-limit, vmax=limit)
    ax.streamplot(vals_x, vals_y, vals_B3x, vals_B3y, color="black", density=1.2, arrowsize=1.0)
    ax.set_title("2. Magnetic Field Lines (Eq 8 / B^3)")
    ax.set_aspect("equal")

    # Plot 3: Paper Replica (Shift Lines) - UNMASKED per request
    ax = axes[1, 0]
    ax.pcolormesh(vals_x, vals_y, vals_alp, cmap="Reds", shading="auto", vmin=0.0, vmax=1.0)
    ax.streamplot(vals_x, vals_y, vals_betax, vals_betay, color="black", density=1.2, arrowsize=1.0)
    ax.contour(vals_x, vals_y, vals_alp, levels=[0.3], colors="k", linewidths=0.8)
    ax.set_title("3. Paper Replica (Shift Streamlines)")
    ax.set_aspect("equal")

    # Plot 4: Shift Magnitude
    Shift_mag = np.sqrt(vals_betax**2 + vals_betay**2)
    ax = axes[1, 1]
    ax.pcolormesh(vals_x, vals_y, Shift_mag, cmap="viridis", shading="auto")
    ax.streamplot(vals_x, vals_y, vals_betax, vals_betay, color="white", density=1.2, arrowsize=1.0)
    ax.set_title("4. Shift Magnitude & Streamlines")
    ax.set_aspect("equal")

    for ax in axes.flat:
        ax.set_xlim(-20, 20); ax.set_ylim(-20, 20)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"🎉 Full Rigorous Output saved to {out_png}")

if __name__ == "__main__":
    main()