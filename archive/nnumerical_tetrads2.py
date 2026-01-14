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

# --------------------------- Grid Fixing --------------------------- #
def get_full_binary_grid(data):
    """
    Removes ghost zones and applies 180-degree rotational symmetry 
    to reconstruct the full binary from the single BH data.
    """
    alp, betax, betay = data["alp"], data["betax"], data["betay"]
    gxx, gxy, gyy = data["gxx"], data["gxy"], data["gyy"]
    x, y = data["x"], data["y"]

    # 1. Slice off Ghost Zones
    start_idx = np.where(x >= -1e-9)[0][0]
    x_right = x[start_idx:]
    
    if alp.shape[1] == len(x):
        axis = 1
        sl = (slice(None), slice(start_idx, None))
    else:
        axis = 0
        sl = (slice(start_idx, None), slice(None))

    alp_r, bx_r, by_r = alp[sl], betax[sl], betay[sl]
    gxx_r, gxy_r, gyy_r = gxx[sl], gxy[sl], gyy[sl]

    # 2. Rotational Symmetry
    alp_l = np.flip(np.flip(alp_r, axis=0), axis=1)
    gxx_l, gxy_l, gyy_l = np.flip(np.flip(gxx_r, axis=0), axis=1), np.flip(np.flip(gxy_r, axis=0), axis=1), np.flip(np.flip(gyy_r, axis=0), axis=1)
    
    bx_l = -np.flip(np.flip(bx_r, axis=0), axis=1)
    by_l = -np.flip(np.flip(by_r, axis=0), axis=1)

    # 3. Stitch
    def stitch(left, right, ax):
        if np.isclose(x_right[0], 0.0):
            sl_drop = [slice(None)] * left.ndim
            sl_drop[ax] = slice(None, -1)
            left_trimmed = left[tuple(sl_drop)]
            return np.concatenate([left_trimmed, right], axis=ax)
        return np.concatenate([left, right], axis=ax)

    if np.isclose(x_right[0], 0.0):
        x_full = np.concatenate([-x_right[:0:-1], x_right]) 
    else:
        x_full = np.concatenate([-x_right[::-1], x_right])
    
    alp_full = stitch(alp_l, alp_r, axis)
    betax_full = stitch(bx_l, bx_r, axis)
    betay_full = stitch(by_l, by_r, axis)
    gxx_full = stitch(gxx_l, gxx_r, axis)
    gxy_full = stitch(gxy_l, gxy_r, axis)
    gyy_full = stitch(gyy_l, gyy_r, axis)
    
    new_data = data.copy()
    new_data.update({
        'x': x_full, 'alp': alp_full, 'betax': betax_full, 'betay': betay_full,
        'gxx': gxx_full, 'gxy': gxy_full, 'gyy': gyy_full,
        'gxz': np.zeros_like(alp_full), 'gyz': np.zeros_like(alp_full), 'gzz': np.ones_like(alp_full),
        'betaz': np.zeros_like(alp_full)
    })
    return new_data

# --------------------------- Geometry Helpers --------------------------- #
def lower_shift(gxx, gxy, gxz, gyy, gyz, gzz, betax, betay, betaz):
    b_x = gxx * betax + gxy * betay + gxz * betaz
    b_y = gxy * betax + gyy * betay + gyz * betaz
    b_z = gxz * betax + gyz * betay + gzz * betaz
    return b_x, b_y, b_z

def coefficients(gxx, gxy, gyy, gxz, gyz):
    # Eq A19 - A21
    A = 1.0 / np.sqrt(gxx)
    det2 = gxx * gyy - gxy**2
    det2 = np.clip(det2, 1e-30, None)
    B = 1.0 / np.sqrt(det2)
    C = gxx * gyz - gxy * gxz
    return A, B, C, det2

def spatial_grads(arr, dx, dy) -> Tuple[np.ndarray, np.ndarray]:
    if arr.shape[1] > 1: 
         d_dy, d_dx = np.gradient(arr, dy, dx, edge_order=2)
    else:
         d_dx, d_dy = np.gradient(arr, dx, dy, edge_order=2)
    return d_dx, d_dy

def get_inverse_metric(data):
    # Computes g^{mu nu} for normalization
    det = data['gxx'] * data['gyy'] - data['gxy']**2
    det_inv = 1.0 / np.clip(det, 1e-20, None)
    
    gamma_uu_xx =  data['gyy'] * det_inv
    gamma_uu_yy =  data['gxx'] * det_inv
    gamma_uu_xy = -data['gxy'] * det_inv
    
    alp = data['alp']
    alp2_inv = 1.0 / np.clip(alp**2, 1e-20, None)
    bx, by = data['betax'], data['betay']
    
    g = {}
    g['tt'] = -alp2_inv
    g['tx'] = bx * alp2_inv
    g['ty'] = by * alp2_inv
    g['tz'] = np.zeros_like(alp)
    
    g['xx'] = gamma_uu_xx - (bx * bx) * alp2_inv
    g['xy'] = gamma_uu_xy - (bx * by) * alp2_inv
    g['yy'] = gamma_uu_yy - (by * by) * alp2_inv
    g['zz'] = 1.0
    
    g['xt'] = g['tx']; g['yt'] = g['ty']; g['zt'] = g['tz']
    g['yx'] = g['xy']
    g['xz'] = np.zeros_like(alp); g['yz'] = np.zeros_like(alp)
    g['zx'] = g['xz']; g['zy'] = g['yz']
    
    return g

def contract_vector_norm(v, g_inv):
    norm_sq = np.zeros_like(v['t'])
    for mu in ['t', 'x', 'y', 'z']:
        for nu in ['t', 'x', 'y', 'z']:
            metric_comp = g_inv[f"{mu}{nu}"] if f"{mu}{nu}" in g_inv else g_inv[f"{nu}{mu}"]
            norm_sq += metric_comp * v[mu] * v[nu]
    return norm_sq

# --------------------------- Tetrad Construction --------------------------- #
def build_tetrads(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    alp = data["alp"]
    gxx, gxy, gyy = data["gxx"], data["gxy"], data["gyy"]
    gxz, gyz, gzz = data["gxz"], data["gyz"], data["gzz"]
    betax, betay, betaz = data["betax"], data["betay"], data["betaz"]

    b_x, b_y, b_z = lower_shift(gxx, gxy, gxz, gyy, gyz, gzz, betax, betay, betaz)
    A, B, C, det2 = coefficients(gxx, gxy, gyy, gxz, gyz)
    B2 = B**2

    g1 = dict(t=b_x, x=gxx, y=gxy, z=gxz)
    g2 = dict(t=b_y, x=gxy, y=gyy, z=gyz)
    g3 = dict(t=b_z, x=gxz, y=gyz, z=gzz)

    theta = {}

    # hat{0} - Eq A15
    theta["0_t"] = alp
    theta["0_x"] = np.zeros_like(alp); theta["0_y"] = np.zeros_like(alp); theta["0_z"] = np.zeros_like(alp)

    # hat{1} - Eq A16
    theta["1_t"] = A * g1["t"]
    theta["1_x"] = A * g1["x"]; theta["1_y"] = A * g1["y"]; theta["1_z"] = A * g1["z"]

    # hat{2} - Eq A17
    theta["2_t"] = A * B * (gxx * g2["t"] - gxy * g1["t"])
    theta["2_x"] = A * B * (gxx * g2["x"] - gxy * g1["x"])
    theta["2_y"] = A * B * (gxx * g2["y"] - gxy * g1["y"])
    theta["2_z"] = A * B * (gxx * g2["z"] - gxy * g1["z"])

    # hat{3} - Corrected from Eq A18 to enforce A13 (Orthonormality)
    # 1. Compute Gram-Schmidt Direction
    raw_3 = {}
    for comp in ["t", "x", "y", "z"]:
        term1 = gxx * g3[comp] - gxz * g1[comp]
        term2 = (gxx * g2[comp] - gxy * g1[comp]) * B2 * C
        raw_3[comp] = term1 - term2

    # 2. Rigorous Normalization
    g_inv = get_inverse_metric(data)
    norm_sq = contract_vector_norm(raw_3, g_inv)
    norm_inv = 1.0 / np.sqrt(np.clip(norm_sq, 1e-20, None))
    
    for comp in ["t", "x", "y", "z"]:
         theta[f"3_{comp}"] = raw_3[comp] * norm_inv

    theta["det2"] = det2
    return theta

# --------------------------- Strict Verification --------------------------- #
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

# --------------------------- Fields & Plotting --------------------------- #
def field_strength(theta, dx, dy):
    F = {k: {} for k in ["0", "1", "2", "3"]}
    for hat in ["0", "1", "2", "3"]:
        dth_t_dx, dth_t_dy = spatial_grads(theta[f"{hat}_t"], dx, dy)
        dth_x_dx, dth_x_dy = spatial_grads(theta[f"{hat}_x"], dx, dy)
        dth_y_dx, dth_y_dy = spatial_grads(theta[f"{hat}_y"], dx, dy)
        F[hat]["F_xt"] = dth_t_dx
        F[hat]["F_yt"] = dth_t_dy
        F[hat]["F_xy"] = dth_y_dx - dth_x_dy
    return F

def gem_fields(theta, F):
    alp = theta["0_t"]
    n_t = 1.0 / np.clip(alp, 1e-30, None)
    E = {}
    for hat in ["0", "1", "2", "3"]:
        E[hat] = (n_t * F[hat]["F_xt"], n_t * F[hat]["F_yt"])
    return E

def shift_curl_bz(betax, betay, dx, dy):
    dbetay_dx, _ = spatial_grads(betay, dx, dy)
    _, dbetax_dy = spatial_grads(betax, dx, dy)
    return dbetay_dx - dbetax_dy

def main(data_path: str = DATA_FILE, out_png: str = "Paper_Method_Tetrads2.png"):
    raw_data = load_data(data_path)
    data = get_full_binary_grid(raw_data)
    x, y = data["x"], data["y"]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])

    print(f"Grid Shape: {data['alp'].shape}")

    theta = build_tetrads(data)
    check_orthonormality(theta, data)

    F = field_strength(theta, dx, dy)
    E = gem_fields(theta, F)
    Bz_phys = shift_curl_bz(data["betax"], data["betay"], dx, dy)

    if data['alp'].shape == (len(y), len(x)):
        vals_alp = data['alp']
        vals_bz = Bz_phys
        vals_Ex = E["0"][0]
        vals_Ey = E["0"][1]
    else:
        vals_alp = data['alp'].T
        vals_bz = Bz_phys.T
        vals_Ex = E["0"][0].T
        vals_Ey = E["0"][1].T

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    im1 = axes[0].pcolormesh(x, y, vals_alp, cmap="Reds", shading="auto", vmin=0.0, vmax=1.0)
    axes[0].streamplot(x, y, vals_Ex, vals_Ey, color="black", density=1.2, arrowsize=1.0)
    axes[0].set_title("Gravito-Electric Field (Dipole)")
    axes[0].set_aspect("equal")
    axes[0].set_xlim(-20, 20); axes[0].set_ylim(-20, 20)
    fig.colorbar(im1, ax=axes[0])

    limit = np.max(np.abs(vals_bz)) * 0.8
    if limit == 0: limit = 0.1
    im2 = axes[1].pcolormesh(x, y, vals_bz, cmap="RdBu_r", shading="auto", vmin=-limit, vmax=limit)
    axes[1].contour(x, y, vals_alp, levels=[0.3], colors="k", linewidths=0.8)
    axes[1].set_title("Gravito-Magnetic Field (Quadrupole)")
    axes[1].set_aspect("equal")
    axes[1].set_xlim(-20, 20); axes[1].set_ylim(-20, 20)
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"🎉 Success. Plot saved to {out_png}")

if __name__ == "__main__":
    main()