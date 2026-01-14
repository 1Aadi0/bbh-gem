import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from typing import Dict

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

def build_tetrad_3(data):
    # Only need Tetrad 3 for rigorous B-field check
    alp = data["alp"]
    gxx, gxy, gyy = data["gxx"], data["gxy"], data["gyy"]
    gxz, gyz, gzz = data["gxz"], data["gyz"], data["gzz"]
    betax, betay, betaz = data["betax"], data["betay"], data["betaz"]

    b_x = gxx*betax + gxy*betay + gxz*betaz
    b_y = gxy*betax + gyy*betay + gyz*betaz
    b_z = gxz*betax + gyz*betay + gzz*betaz
    
    A, B, C, det2 = coefficients(gxx, gxy, gyy, gxz, gyz)
    B2 = B**2

    g1 = dict(t=b_x, x=gxx, y=gxy, z=gxz)
    g2 = dict(t=b_y, x=gxy, y=gyy, z=gyz)
    g3 = dict(t=b_z, x=gxz, y=gyz, z=gzz)

    raw_3 = {}
    for comp in ["t", "x", "y", "z"]:
        term1 = gxx * g3[comp] - gxz * g1[comp]
        term2 = (gxx * g2[comp] - gxy * g1[comp]) * B2 * C
        raw_3[comp] = term1 - term2

    g_inv = get_inverse_metric(data)
    norm_sq = contract_vector_norm(raw_3, g_inv)
    norm_inv = 1.0 / np.sqrt(np.clip(norm_sq, 1e-20, None))
    
    theta3 = {}
    for comp in ["t", "x", "y", "z"]:
         theta3[comp] = raw_3[comp] * norm_inv
         
    return theta3

# --------------------------- Calculations --------------------------- #
def calculate_B3_via_eq8(theta3, dx, dy):
    Az = theta3["z"] 
    if Az.shape[0] > Az.shape[1]: # (nx, ny)
        dAz_dx, dAz_dy = np.gradient(Az, dx, dy, edge_order=2)
    else: # (ny, nx)
        dAz_dy, dAz_dx = np.gradient(Az, dy, dx, edge_order=2)
        
    B3_x = dAz_dy
    B3_y = -dAz_dx
    return B3_x, B3_y

def shift_curl_magnitude(betax, betay, dx, dy):
    if betax.shape[0] > betax.shape[1]: 
         dby_dx, dby_dy = np.gradient(betay, dx, dy, edge_order=2)
         dbx_dx, dbx_dy = np.gradient(betax, dx, dy, edge_order=2)
    else: 
         dby_dy, dby_dx = np.gradient(betay, dy, dx, edge_order=2)
         dbx_dy, dbx_dx = np.gradient(betax, dy, dx, edge_order=2)
    return dby_dx - dbx_dy

def get_electric_field(alp, dx, dy):
    # Smooth the lapse to fix center seam artifacts
    alp_smooth = gaussian_filter(alp, sigma=1.5)
    
    if alp.shape[0] > alp.shape[1]: # (nx, ny)
        grad_x, grad_y = np.gradient(alp_smooth, dx, dy)
    else:
        grad_y, grad_x = np.gradient(alp_smooth, dy, dx)
        
    return -grad_x, -grad_y

# --------------------------- Main Plotting --------------------------- #
def main(data_path: str = DATA_FILE, out_png: str = "Comparison_Quad_Plot.png"):
    raw_data = load_data(data_path)
    data = get_full_binary_grid(raw_data)
    x, y = data["x"], data["y"]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])

    # --- 1. Compute Fields ---
    # A. Electric Field
    Ex, Ey = get_electric_field(data["alp"], dx, dy)
    
    # B. Magnetic Field (Rigorous Eq 8)
    theta3 = build_tetrad_3(data)
    B3_x, B3_y = calculate_B3_via_eq8(theta3, dx, dy)
    
    # C. Shift Vector & Curl
    Bz_mag = shift_curl_magnitude(data["betax"], data["betay"], dx, dy)
    Shift_mag = np.sqrt(data["betax"]**2 + data["betay"]**2)

    # --- 2. Prep for Plotting (Rotate all to Y, X) ---
    def prep(arr):
        if arr.shape == (len(y), len(x)): return arr
        return arr.T

    # Coordinates
    vals_x, vals_y = x, y
    
    # Scalar Backgrounds
    vals_alp = prep(data['alp'])
    vals_Bz_mag = prep(Bz_mag)
    vals_Shift_mag = prep(Shift_mag)
    
    # Vectors (Streamlines)
    vals_Ex, vals_Ey = prep(Ex), prep(Ey)
    vals_B3x, vals_B3y = prep(B3_x), prep(B3_y)
    vals_betax, vals_betay = prep(data['betax']), prep(data['betay'])

    # --- 3. Plotting Grid ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Limits for colors
    mag_lim = np.max(np.abs(vals_Bz_mag)) * 0.8
    
    # Plot 1: Electric Field Lines (Standard)
    ax = axes[0, 0]
    ax.pcolormesh(vals_x, vals_y, vals_alp, cmap="Reds", shading="auto", vmin=0.0, vmax=1.0)
    ax.streamplot(vals_x, vals_y, vals_Ex, vals_Ey, color="black", density=1.2, arrowsize=1.0)
    ax.set_title("1. Electric Field Lines (on Lapse)")
    ax.set_aspect("equal")
    
    # Plot 2: Magnetic Field Lines (Eq 8) + Mag Strength
    ax = axes[0, 1]
    ax.pcolormesh(vals_x, vals_y, vals_Bz_mag, cmap="RdBu_r", shading="auto", vmin=-mag_lim, vmax=mag_lim)
    ax.streamplot(vals_x, vals_y, vals_B3x, vals_B3y, color="black", density=1.2, arrowsize=1.0)
    ax.set_title("2. Eq. 8 Magnetic Lines (on Mag Strength)")
    ax.set_aspect("equal")

    # Plot 3: Shift Vector Lines + Mag Strength (PAPER MATCH?)
    ax = axes[1, 0]
    ax.pcolormesh(vals_x, vals_y, vals_Bz_mag, cmap="RdBu_r", shading="auto", vmin=-mag_lim, vmax=mag_lim)
    ax.streamplot(vals_x, vals_y, vals_betax, vals_betay, color="black", density=1.2, arrowsize=1.0)
    ax.set_title("3. Shift Streamlines (on Mag Strength)")
    ax.set_aspect("equal")
    
    # Plot 4: Shift Vector Lines + Shift Magnitude
    ax = axes[1, 1]
    ax.pcolormesh(vals_x, vals_y, vals_Shift_mag, cmap="viridis", shading="auto")
    ax.streamplot(vals_x, vals_y, vals_betax, vals_betay, color="white", density=1.2, arrowsize=1.0)
    ax.set_title("4. Shift Streamlines (on Shift Magnitude)")
    ax.set_aspect("equal")

    for ax in axes.flat:
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"🎉 Comparison Grid saved to {out_png}")

if __name__ == "__main__":
    main()