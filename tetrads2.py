import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple, Optional

DATA_FILE = "final_data.npz"

def load_data(file_path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}")
    d = np.load(file_path)
    # Extract
    alp = d["alp"]
    gxx, gxy, gyy = d["gxx"], d["gxy"], d["gyy"]
    gxz = d.get("gxz", np.zeros_like(gxx))
    gyz = d.get("gyz", np.zeros_like(gxx))
    gzz = d.get("gzz", np.ones_like(gxx))
    betax, betay = d["betax"], d["betay"]
    betaz = d.get("betaz", np.zeros_like(betax))
    x, y = d["x"], d["y"]
    
    return dict(alp=alp, gxx=gxx, gxy=gxy, gyy=gyy, gxz=gxz, gyz=gyz, gzz=gzz,
                betax=betax, betay=betay, betaz=betaz, x=x, y=y)

def lower_shift(gxx, gxy, gxz, gyy, gyz, gzz, betax, betay, betaz):
    b_x = gxx * betax + gxy * betay + gxz * betaz
    b_y = gxy * betax + gyy * betay + gyz * betaz
    b_z = gxz * betax + gyz * betay + gzz * betaz
    return b_x, b_y, b_z

def coefficients(gxx, gxy, gyy, gxz, gyz):
    A = 1.0 / np.sqrt(gxx)
    det2 = gxx * gyy - gxy**2
    det2 = np.clip(det2, 1e-30, None)
    B = 1.0 / np.sqrt(det2)
    C = gxx * gyz - gxy * gxz
    return A, B, C, det2

def spatial_grads(arr, dx, dy) -> Tuple[np.ndarray, np.ndarray]:
    # CRITICAL FIX: Data shape is (ny, nx) -> (123, 57).
    # Axis 0 is Y (spacing dy). Axis 1 is X (spacing dx).
    # np.gradient(arr, h0, h1) -> returns (grad_axis0, grad_axis1)
    grad_y, grad_x = np.gradient(arr, dy, dx, edge_order=2)
    return grad_x, grad_y

def build_tetrads(data):
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
    theta["0_t"] = alp
    theta["0_x"] = np.zeros_like(alp); theta["0_y"] = np.zeros_like(alp); theta["0_z"] = np.zeros_like(alp)
    theta["1_t"] = A * g1["t"]; theta["1_x"] = A * g1["x"]; theta["1_y"] = A * g1["y"]; theta["1_z"] = A * g1["z"]
    theta["2_t"] = A * B * (gxx * g2["t"] - gxy * g1["t"])
    theta["2_x"] = A * B * (gxx * g2["x"] - gxy * g1["x"])
    theta["2_y"] = A * B * (gxx * g2["y"] - gxy * g1["y"])
    theta["2_z"] = A * B * (gxx * g2["z"] - gxy * g1["z"])
    
    for comp, g3nu, g2nu, g1nu in [("t", g3["t"], g2["t"], g1["t"]), ("x", g3["x"], g2["x"], g1["x"]), 
                                   ("y", g3["y"], g2["y"], g1["y"]), ("z", g3["z"], g2["z"], g1["z"])]:
        term1 = gxx * g3nu - gxz * g1nu
        term2 = (gxx * g2nu - gxy * g1nu) * B2 * C
        theta[f"3_{comp}"] = A * (term1 - term2)

    theta["det2"] = det2
    return theta

def field_strength(theta, dx, dy):
    F = {k: {} for k in ["0", "1", "2", "3"]}
    for hat in ["0", "1", "2", "3"]:
        th_t, th_x, th_y = theta[f"{hat}_t"], theta[f"{hat}_x"], theta[f"{hat}_y"]
        dth_t_dx, dth_t_dy = spatial_grads(th_t, dx, dy)
        dth_x_dx, dth_x_dy = spatial_grads(th_x, dx, dy)
        dth_y_dx, dth_y_dy = spatial_grads(th_y, dx, dy)

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
    Bz = {hat: F[hat]["F_xy"] for hat in ["0", "1", "2", "3"]}
    return E, Bz

def shift_curl_bz(betax, betay, dx, dy):
    dbetay_dx, _ = spatial_grads(betay, dx, dy)
    _, dbetax_dy = spatial_grads(betax, dx, dy)
    return dbetay_dx - dbetax_dy

def main(data_path=DATA_FILE, out_png="Paper_Method_Tetrads.png"):
    print(f"🚀 Loading data from {data_path}...")
    try:
        data = load_data(data_path)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    x, y = data["x"], data["y"]
    # Usually coordinate arrays are 1D. 
    # Check if x, y are strictly 1D or meshgrids. Assuming 1D here.
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    print("--> Building Tetrads & Fields...")
    theta = build_tetrads(data)
    F = field_strength(theta, dx, dy)
    E, Bz = gem_fields(theta, F)
    Bz_shift = shift_curl_bz(data["betax"], data["betay"], dx, dy)

    print("Grid shape:", data["alp"].shape)
    
    # --- PLOTTING FIX ---
    # Data shape is (ny, nx) = (123, 57)
    # Grid X (ij) is (57, 123) -> X.T is (123, 57)
    # We must plot X.T, Y.T vs Data (no transpose)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Ex0, Ey0 = E["0"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Electric Field
    # Note: data["alp"] is NOT transposed. X and Y ARE transposed.
    im1 = axes[0].pcolormesh(X.T, Y.T, data["alp"], cmap="Reds", shading="auto", vmin=0.0, vmax=1.0)
    axes[0].streamplot(X.T, Y.T, Ex0, Ey0, color="black", density=1.5, arrowsize=1.2)
    axes[0].set_title(r"Electric Field $E^{\hat{0}}$")
    axes[0].set_aspect("equal")
    fig.colorbar(im1, ax=axes[0], label=r"$\alpha$")

    # Plot 2: Magnetic Field
    limit = np.max(np.abs(Bz_shift)) * 0.8
    if limit == 0: limit = 0.1
    im2 = axes[1].pcolormesh(X.T, Y.T, Bz_shift, cmap="RdBu_r", shading="auto", vmin=-limit, vmax=limit)
    try:
        cs = axes[1].contour(X.T, Y.T, Bz_shift, colors="k", linewidths=0.8, levels=15)
        axes[1].clabel(cs, inline=True, fontsize=8)
    except: pass
    
    axes[1].set_title(r"Magnetic Field (curl $\beta$)")
    axes[1].set_aspect("equal")
    fig.colorbar(im2, ax=axes[1], label=r"$B_z$")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"🎉 Saved {out_png}")

if __name__ == "__main__":
    main()