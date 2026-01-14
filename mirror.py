import numpy as np
import matplotlib.pyplot as plt
import os

DATA_FILE = "final_data.npz"

def apply_rotational_symmetry():
    print("🚀 APPLYING 180-DEGREE ROTATIONAL SYMMETRY...")
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: {DATA_FILE} not found.")
        return

    # 1. Load Data
    d = np.load(DATA_FILE)
    alp = d['alp']
    betax, betay = d['betax'], d['betay']
    x, y = d['x'], d['y']
    
    print(f"   Raw Grid X: [{x[0]:.2f}, ..., {x[-1]:.2f}]")

    # 2. Slice off Ghost Zones (Keep x >= 0)
    valid_indices = np.where(x >= -1e-10)[0] # Find index of x=0
    start_idx = valid_indices[0]
    
    x_clean = x[start_idx:]
    # Slice arrays (Axis 1 is X)
    alp_right = alp[:, start_idx:]
    betax_right = betax[:, start_idx:]
    betay_right = betay[:, start_idx:]

    # 3. Create Left Side (Rotational Symmetry)
    # Rule: Data(-x, y) = Data(x, -y)
    # Operation: Flip Horizontal AND Flip Vertical
    
    def rotate_180(arr, flip_sign=False):
        # Flip X (Axis 1) AND Flip Y (Axis 0)
        arr_rot = arr[::-1, ::-1]
        
        # Vectors flip direction under 180-rotation
        if flip_sign:
            arr_rot = -arr_rot
            
        return arr_rot

    # Generate Left Data
    # For Shift Vector: Rotation flips the vector direction -> sign change
    alp_left = rotate_180(alp_right, flip_sign=False)   # Scalar: No sign change
    betax_left = rotate_180(betax_right, flip_sign=True) # Vector: Sign change
    betay_left = rotate_180(betay_right, flip_sign=True) # Vector: Sign change

    # 4. Stitch (Handle x=0 overlap)
    # If x starts exactly at 0, we drop the last column of the Left side to avoid duplication
    if np.isclose(x_clean[0], 0.0):
        # Create X axis [-x_n ... -x_1, 0, x_1 ... x_n]
        x_left_axis = -x_clean[:0:-1] 
        x_full = np.concatenate([x_left_axis, x_clean])
        
        # Slice off x=0 column from left data
        alp_full = np.concatenate([alp_left[:, :-1], alp_right], axis=1)
        betax_full = np.concatenate([betax_left[:, :-1], betax_right], axis=1)
        betay_full = np.concatenate([betay_left[:, :-1], betay_right], axis=1)
    else:
        # Simple concat
        x_left_axis = -x_clean[::-1]
        x_full = np.concatenate([x_left_axis, x_clean])
        
        alp_full = np.concatenate([alp_left, alp_right], axis=1)
        betax_full = np.concatenate([betax_left, betax_right], axis=1)
        betay_full = np.concatenate([betay_left, betay_right], axis=1)

    print(f"   Final Grid X: [{x_full[0]:.2f}, ..., {x_full[-1]:.2f}]")

    # 5. Physics (Recalculate on Full Grid)
    # Now that the vector field is smooth across x=0, the derivatives will be correct.
    dx = x_full[1] - x_full[0]
    dy = y[1] - y[0]
    
    # B_z ~ d_x(beta_y) - d_y(beta_x)
    dby_dx = np.gradient(betay_full, dx, axis=1)
    dbx_dy = np.gradient(betax_full, dy, axis=0)
    Bz = dby_dx - dbx_dy
    
    # E ~ Grad(Lapse)
    grad_y, grad_x = np.gradient(alp_full, dy, dx)
    Ex, Ey = -grad_x, -grad_y

    # 6. Plot
    print("--> Generating Corrected Plot...")
    X, Y = np.meshgrid(x_full, y, indexing='ij')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Electric Field
    ax1 = axes[0]
    # Note: No Transpose (.T) on data because we match shapes now
    im1 = ax1.pcolormesh(X.T, Y.T, alp_full, cmap='Reds', shading='auto', vmin=0.0, vmax=1.0)
    ax1.streamplot(X.T, Y.T, Ex, Ey, color='black', density=1.0, arrowsize=1.0)
    ax1.set_title("Gravito-Electric Field (Dipole)")
    ax1.set_xlim(-20, 20); ax1.set_ylim(-20, 20)
    ax1.set_aspect('equal')
    
    # Magnetic Field
    ax2 = axes[1]
    limit = np.max(np.abs(Bz)) * 0.8
    if limit == 0: limit = 0.1
    
    im2 = ax2.pcolormesh(X.T, Y.T, Bz, cmap='RdBu_r', shading='auto', vmin=-limit, vmax=limit)
    ax2.contour(X.T, Y.T, alp_full, levels=[0.3], colors='black', linewidths=0.5)
    ax2.set_title("Gravito-Magnetic Field (Quadrupole)")
    ax2.set_xlim(-20, 20); ax2.set_ylim(-20, 20)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig("Final_Rotational_Symmetry.png", dpi=150)
    print("🎉 SUCCESS. Open 'Final_Rotational_Symmetry.png'")

if __name__ == "__main__":
    apply_rotational_symmetry()