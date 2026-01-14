import numpy as np
import matplotlib.pyplot as plt
import os

DATA_FILE = "final_data.npz"

def mirror_data(data_dict):
    """Mirrors data across x=0 if the grid starts at x=0."""
    x = data_dict['x']
    if x[0] >= -0.1: 
        print("   Detected Symmetry Boundary at x=0. Mirroring data...")
        x_left = -x[::-1]
        x_new = np.concatenate([x_left[:-1], x])
        
        new_data = {'x': x_new, 'y': data_dict['y']}
        
        for key, arr in data_dict.items():
            if key in ['x', 'y']: continue
            if arr.ndim == 2:
                # Mirror horizontally
                arr_left = arr[:, ::-1]
                # Flip vector x-components
                if key in ['betax', 'gxy', 'gxz']: 
                    arr_left = -arr_left
                arr_new = np.concatenate([arr_left[:, :-1], arr], axis=1)
                new_data[key] = arr_new
        return new_data
    else:
        return data_dict

def main():
    if not os.path.exists(DATA_FILE): 
        print(f"❌ Error: {DATA_FILE} not found.")
        return

    # 1. Load & Mirror
    d = np.load(DATA_FILE)
    data = dict(d)
    data = mirror_data(data)
    
    alp = data['alp']
    betax, betay = data['betax'], data['betay']
    x, y = data['x'], data['y']
    
    print(f"   Grid Shape: {alp.shape}")
    
    # 2. Physics
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # B_z ~ Curl(Shift)
    dby_dx = np.gradient(betay, dx, axis=1)
    dbx_dy = np.gradient(betax, dy, axis=0)
    Bz = dby_dx - dbx_dy
    
    # E ~ Grad(Lapse)
    grad_y, grad_x = np.gradient(alp, dy, dx)
    Ex, Ey = -grad_x, -grad_y
    
    # 3. Plotting
    print("--> Generating Plot...")
    
    # Use 'ij' indexing. X shape is (nx, ny).
    # We transpose X, Y to get (ny, nx) which matches alp.
    X, Y = np.meshgrid(x, y, indexing='ij') 
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # --- PLOT 1: ELECTRIC FIELD ---
    ax1 = axes[0]
    # FIX: Use 'alp' directly (do NOT transpose it)
    im1 = ax1.pcolormesh(X.T, Y.T, alp, cmap='Reds', shading='auto', vmin=0.0, vmax=1.0)
    # FIX: Use 'Ex', 'Ey' directly
    ax1.streamplot(X.T, Y.T, Ex, Ey, color='black', density=1.2, arrowsize=1.0)
    ax1.set_title("Electric Field (Lapse Gradient)")
    ax1.set_aspect('equal')
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(y[0], y[-1])
    
    # --- PLOT 2: MAGNETIC FIELD ---
    ax2 = axes[1]
    limit = np.max(np.abs(Bz)) * 0.8
    if limit == 0: limit = 0.1
    
    # FIX: Use 'Bz' directly
    im2 = ax2.pcolormesh(X.T, Y.T, Bz, cmap='RdBu_r', shading='auto', vmin=-limit, vmax=limit)
    # FIX: Use 'alp' directly for contours
    ax2.contour(X.T, Y.T, alp, levels=[0.3], colors='black', linewidths=0.5)
    ax2.set_title("Magnetic Field (Shift Curl)")
    ax2.set_aspect('equal')
    ax2.set_xlim(x[0], x[-1])
    ax2.set_ylim(y[0], y[-1])
    
    plt.tight_layout()
    plt.savefig("Full_Binary_Plot_Fixed.png", dpi=150)
    print("🎉 Saved Full_Binary_Plot_Fixed.png")

if __name__ == "__main__":
    main()