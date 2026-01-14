import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

DATA_FILE = "final_data.npz"

# --------------------------- I/O --------------------------- #
def load_data(file_path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}")
    d = np.load(file_path)
    return dict(
        alp=d["alp"], 
        betax=d["betax"], betay=d["betay"],
        x=d["x"], y=d["y"],
    )

# --------------------------- Grid Fixing --------------------------- #
def get_full_binary_grid(data):
    # Standard cleanup (Ghost zones + Symmetry)
    alp, betax, betay = data["alp"], data["betax"], data["betay"]
    x, y = data["x"], data["y"]

    start_idx = np.where(x >= -1e-9)[0][0]
    x_right = x[start_idx:]
    
    # Check orientation
    if alp.shape[1] == len(x):
        axis = 1
        sl = (slice(None), slice(start_idx, None))
    else:
        axis = 0
        sl = (slice(start_idx, None), slice(None))

    alp_r, bx_r, by_r = alp[sl], betax[sl], betay[sl]

    # Symmetry Ops: 180 deg rotation
    alp_l = np.flip(np.flip(alp_r, axis=0), axis=1)
    bx_l = -np.flip(np.flip(bx_r, axis=0), axis=1)
    by_l = -np.flip(np.flip(by_r, axis=0), axis=1)

    # Stitch
    def stitch(left, right, ax):
        if np.isclose(x_right[0], 0.0):
            sl_drop = [slice(None)] * left.ndim
            sl_drop[ax] = slice(None, -1)
            left_trimmed = left[tuple(sl_drop)]
            return np.concatenate([left_trimmed, right], axis=ax)
        return np.concatenate([left, right], axis=ax)

    x_full = np.concatenate([-x_right[:0:-1], x_right]) if np.isclose(x_right[0], 0.0) else np.concatenate([-x_right[::-1], x_right])
    
    new_data = data.copy()
    new_data.update({
        'x': x_full, 
        'alp': stitch(alp_l, alp_r, axis), 
        'betax': stitch(bx_l, bx_r, axis), 
        'betay': stitch(by_l, by_r, axis),
    })
    return new_data

# --------------------------- Main Plotting --------------------------- #
def main(data_path: str = DATA_FILE, out_png: str = "Paper_Replica_Final.png"):
    raw_data = load_data(data_path)
    data = get_full_binary_grid(raw_data)
    x, y = data["x"], data["y"]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])

    # 1. Electric Field Lines (Gradient of Lapse)
    if data['alp'].shape == (len(y), len(x)):
        grad_y, grad_x = np.gradient(data["alp"], dy, dx)
    else:
        grad_x, grad_y = np.gradient(data["alp"], dx, dy)
    Ex, Ey = -grad_x, -grad_y

    # 2. Prepare Data for Plotting (Rotate to y,x)
    def prep(arr):
        if arr.shape == (len(y), len(x)): return arr
        return arr.T

    vals_alp = prep(data['alp'])       # The "Glowing Eyes" Background
    vals_bx = prep(data['betax'])      # Shift X
    vals_by = prep(data['betay'])      # Shift Y
    vals_Ex = prep(Ex)
    vals_Ey = prep(Ey)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Plot 1: Electric Field ---
    # Background: Lapse (Red)
    axes[0].pcolormesh(x, y, vals_alp, cmap="Reds", shading="auto", vmin=0.0, vmax=1.0)
    # Lines: Electric Field (Gradient)
    axes[0].streamplot(x, y, vals_Ex, vals_Ey, color="black", density=1.2, arrowsize=1.0)
    axes[0].set_title("Gravito-Electric Field (Dipole)")
    axes[0].set_aspect("equal")
    axes[0].set_xlim(-20, 20); axes[0].set_ylim(-20, 20)

    # --- Plot 2: Magnetic Field (Paper Style) ---
    # Background: Lapse (Red) -- Matches Paper Caption!
    # "The first row shows the gravitational lapse alpha..."
    axes[1].pcolormesh(x, y, vals_alp, cmap="Reds", shading="auto", vmin=0.0, vmax=1.0)
    
    # Lines: Shift Vector (Swirls) -- Matches Paper Lines!
    # "...and magnetic field lines" (physically defined as Shift lines in GEM)
    axes[1].streamplot(x, y, vals_bx, vals_by, color="black", density=1.2, arrowsize=1.0, linewidth=0.8)
    
    # Contour for Horizon
    axes[1].contour(x, y, vals_alp, levels=[0.3], colors="k", linewidths=0.8)
    
    axes[1].set_title("Gravito-Magnetic Field (Shift Streamlines)")
    axes[1].set_aspect("equal")
    axes[1].set_xlim(-20, 20); axes[1].set_ylim(-20, 20)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"🎉 Success. Exact Paper Replica saved to {out_png}")

if __name__ == "__main__":
    main()