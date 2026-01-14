import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

DATA_FILE = "final_data.npz"

# --------------------------- I/O & Grid --------------------------- #
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

def get_full_binary_grid(data):
    # Standard Grid Reconstruction (Ghosts + Symmetry)
    alp, betax, betay = data["alp"], data["betax"], data["betay"]
    gxx, gxy, gyy = data["gxx"], data["gxy"], data["gyy"]
    x, y = data["x"], data["y"]

    start_idx = np.where(x >= -1e-9)[0][0]
    x_right = x[start_idx:]
    axis = 1 if alp.shape[1] == len(x) else 0
    sl = (slice(None), slice(start_idx, None)) if axis == 1 else (slice(start_idx, None), slice(None))

    alp_r, bx_r, by_r = alp[sl], betax[sl], betay[sl]
    gxx_r, gxy_r, gyy_r = gxx[sl], gxy[sl], gyy[sl]
    gzz_r = data["gzz"][sl] # Grab gzz too

    # Symmetry Ops
    alp_l = np.flip(np.flip(alp_r, axis=0), axis=1)
    gxx_l = np.flip(np.flip(gxx_r, axis=0), axis=1)
    gxy_l = np.flip(np.flip(gxy_r, axis=0), axis=1)
    gyy_l = np.flip(np.flip(gyy_r, axis=0), axis=1)
    gzz_l = np.flip(np.flip(gzz_r, axis=0), axis=1)
    
    bx_l = -np.flip(np.flip(bx_r, axis=0), axis=1)
    by_l = -np.flip(np.flip(by_r, axis=0), axis=1)

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

# --------------------------- Math & Tetrads --------------------------- #
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
    g['zz'] = 1.0 / np.clip(data['gzz'], 1e-20, None) # Simple inverse for z
    
    # 4D Inverse terms (needed for norm)
    alp = data['alp']
    alp2_inv = 1.0 / np.clip(alp**2, 1e-20, None)
    bx, by = data['betax'], data['betay']
    
    g['tt'] = -alp2_inv
    g['tx'] = bx * alp2_inv
    g['ty'] = by * alp2_inv
    g['tz'] = np.zeros_like(alp)
    
    # Correct g^ij = gamma^ij - beta^i beta^j / alpha^2
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

def build_tetrad_3(data):
    # We only need Tetrad 3 for the magnetic lines
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

    # Standard basis 1-forms
    g1 = dict(t=b_x, x=gxx, y=gxy, z=gxz)
    g2 = dict(t=b_y, x=gxy, y=gyy, z=gyz)
    g3 = dict(t=b_z, x=gxz, y=gyz, z=gzz)

    # Hat 3 Direction (Paper Eq A18 numerator)
    raw_3 = {}
    for comp in ["t", "x", "y", "z"]:
        term1 = gxx * g3[comp] - gxz * g1[comp]
        term2 = (gxx * g2[comp] - gxy * g1[comp]) * B2 * C
        raw_3[comp] = term1 - term2

    # Normalization
    g_inv = get_inverse_metric(data)
    norm_sq = contract_vector_norm(raw_3, g_inv)
    norm_inv = 1.0 / np.sqrt(np.clip(norm_sq, 1e-20, None))
    
    theta3 = {}
    for comp in ["t", "x", "y", "z"]:
         theta3[comp] = raw_3[comp] * norm_inv
         
    return theta3

# --------------------------- Equation 8 Implementation --------------------------- #
def calculate_B3_via_eq8(theta3, dx, dy):
    """
    Calculates B^3_i = epsilon^ijk d_j A_k using Eq 8.
    We assume symmetry in z (d_z = 0) for the 2D slice.
    """
    # Components of A^3
    Az = theta3["z"] # This is the dominant component
    
    # B^3_x = d_y A_z - d_z A_y  (assume d_z=0) -> d_y A_z
    # B^3_y = d_z A_x - d_x A_z  (assume d_z=0) -> -d_x A_z
    
    # Note: We handle (nx, ny) vs (ny, nx) carefully
    if Az.shape[0] > Az.shape[1]: # (nx, ny)
        dAz_dx, dAz_dy = np.gradient(Az, dx, dy, edge_order=2)
    else: # (ny, nx)
        dAz_dy, dAz_dx = np.gradient(Az, dy, dx, edge_order=2)
        
    B3_x = dAz_dy
    B3_y = -dAz_dx
    
    return B3_x, B3_y

def shift_curl_bz_magnitude(betax, betay, dx, dy):
    # This is still the correct physical scalar for the COLOR plot (Frame Dragging Strength)
    if betax.shape[0] > betax.shape[1]: 
         dby_dx, dby_dy = np.gradient(betay, dx, dy, edge_order=2)
         dbx_dx, dbx_dy = np.gradient(betax, dx, dy, edge_order=2)
    else: 
         dby_dy, dby_dx = np.gradient(betay, dy, dx, edge_order=2)
         dbx_dy, dbx_dx = np.gradient(betax, dy, dx, edge_order=2)
    return dby_dx - dbx_dy

# --------------------------- Main --------------------------- #
def main(data_path: str = DATA_FILE, out_png: str = "Paper_Method_Rigorous.png"):
    raw_data = load_data(data_path)
    data = get_full_binary_grid(raw_data)
    x, y = data["x"], data["y"]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])

    print(f"Grid Shape: {data['alp'].shape}")

    # 1. Build Spatial Tetrad 3 (The z-leg)
    theta3 = build_tetrad_3(data)

    # 2. Calculate B^3 vector using Equation 8 (RIGOROUS!)
    B3_x, B3_y = calculate_B3_via_eq8(theta3, dx, dy)
    
    # 3. Calculate Scalar for Color (Shift Curl / GEM B)
    # The paper likely colors by the "GEM" field strength (Bz) but plots lines of B^3
    Bz_color = shift_curl_bz_magnitude(data["betax"], data["betay"], dx, dy)

    # Plot Prep
    def prep(arr):
        if arr.shape == (len(y), len(x)): return arr
        return arr.T

    vals_alp = prep(data['alp'])
    vals_bz_color = prep(Bz_color)
    vals_b3_x = prep(B3_x)
    vals_b3_y = prep(B3_y)
    
    # Electric field (Simple gradient of lapse for context)
    grad_y, grad_x = np.gradient(data["alp"], dy, dx) if data['alp'].shape == (len(y), len(x)) else np.gradient(data["alp"], dx, dy)
    vals_Ex, vals_Ey = prep(-grad_x), prep(-grad_y)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Electric Field ---
    axes[0].pcolormesh(x, y, vals_alp, cmap="Reds", shading="auto", vmin=0.0, vmax=1.0)
    axes[0].streamplot(x, y, vals_Ex, vals_Ey, color="black", density=1.2, arrowsize=1.0)
    axes[0].set_title("Gravito-Electric Field (Dipole)")
    axes[0].set_aspect("equal")
    axes[0].set_xlim(-20, 20); axes[0].set_ylim(-20, 20)

    # --- Magnetic Field (Strictly Eq 8) ---
    limit = np.max(np.abs(vals_bz_color)) * 0.8 or 0.1
    axes[1].pcolormesh(x, y, vals_bz_color, cmap="RdBu_r", shading="auto", vmin=-limit, vmax=limit)
    
    # LINES are now B^3 (Spatial Tetrad Magnetic Field)
    # This forms the toroidal loops strictly from tetrad derivatives!
    axes[1].streamplot(x, y, vals_b3_x, vals_b3_y, color="black", density=1.2, arrowsize=1.0, linewidth=0.8)
    
    axes[1].contour(x, y, vals_alp, levels=[0.3], colors="k", linewidths=0.8)
    axes[1].set_title("Gravito-Magnetic Field (Spatial Tetrad B^3)")
    axes[1].set_aspect("equal")
    axes[1].set_xlim(-20, 20); axes[1].set_ylim(-20, 20)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"🎉 Success. Plot saved to {out_png}")

if __name__ == "__main__":
    main()