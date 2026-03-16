import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify

def calculate_automated_fields():
    # # --- 1. Variables and Metric Setup ---
    # t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    # M = sp.symbols('M', real=True, positive=True)
    # R_far = sp.symbols('R', real=True, positive=True) # For display in Far Field
    # rs = 2 * M
    
    # Use exact fractions to prevent floating point pollution
    half = sp.Rational(1, 2)
    
    # alpha_func = sp.sqrt(1 - rs/r)
    
    # # Covariant Metric g_uv
    # g_cov = sp.diag(-(alpha_func**2), 1/(alpha_func**2), r**2, r**2 * sp.sin(theta)**2)
    # g_inv = g_cov.inv()
    
    # # Fix the Absolute Value Branch Cut for Spherical Coordinates
    # sqrt_det_g = sp.simplify(sp.sqrt(-g_cov.det())).replace(sp.Abs(sp.sin(theta)), sp.sin(theta))
    
    # # Non-Rotated Tetrad Legs A_mu^(hat_a)
    # Tetrad = sp.Matrix([
    #     [alpha_func, 0, 0, 0],           # t-leg (index 0)
    #     [0, 1/alpha_func, 0, 0],         # r-leg (index 1)
    #     [0, 0, r, 0],                    # theta-leg (index 2)
    #     [0, 0, 0, r*sp.sin(theta)]       # phi-leg (index 3)
    # ])

    # print("\n=== 1. NON-ROTATED TETRAD MATRIX (Diagonal / Spherical) ===")
    # print("Rows: [t, r, theta, phi]_hat | Cols: [dt, dr, dtheta, dphi]")
    # sp.pprint(Tetrad)

    t, x, y, z = sp.symbols('t x y z', real=True)
    M = sp.symbols('M', real=True, positive=True)
    
    half = sp.Rational(1, 2)
    
    # Isotropic radial coordinate
    r_bar = sp.sqrt(x**2 + y**2 + z**2)
    
    # Conformal factor
    psi = 1 + M / (2 * r_bar)
    
    # Lapse function
    alpha_func = (1 - M / (2 * r_bar)) / psi
    
    # Covariant Metric g_uv (Cartesian)
    # g_tt = -alpha^2, g_xx = g_yy = g_zz = psi^4
    g_cov = sp.diag(
        -(alpha_func**2), 
        psi**4, 
        psi**4, 
        psi**4
    )
    g_inv = g_cov.inv()
    
    # Determinant of the metric: sqrt(-g) = alpha * psi^6
    # We define it manually here to prevent SymPy from getting stuck 
    # taking the square root of massive Cartesian polynomials.
    sqrt_det_g = sp.simplify(alpha_func * psi**6)
    
    # Non-Rotated Tetrad Legs A_mu^(hat_a)
    # In Cartesian, the tetrad maps cleanly to [t, x, y, z]
    Tetrad = sp.Matrix([
        [alpha_func, 0, 0, 0],           # t-leg (index 0) maps to dt
        [0, psi**2, 0, 0],               # x-leg (index 1) maps to dx
        [0, 0, psi**2, 0],               # y-leg (index 2) maps to dy
        [0, 0, 0, psi**2]                # z-leg (index 3) maps to dz
    ])
    
    n_cov = sp.Matrix([-alpha_func, 0, 0, 0])
    
    # UPDATE COORDINATE ARRAY
    coords = [t, x, y, z]

    # # #Test case
    # alpha_func = 1
    # # # Covariant Metric g_uv
    # g_cov = sp.diag(-1, 1, r**2, r**2 * sp.sin(theta)**2)
    # g_inv = g_cov.inv()
    # # Fix the Absolute Value Branch Cut for Spherical Coordinates
    # sqrt_det_g = sp.simplify(sp.sqrt(-g_cov.det())).replace(sp.Abs(sp.sin(theta)), sp.sin(theta))
    
    
    # # Non-Rotated Tetrad Legs A_mu^(hat_a)
    # Tetrad = sp.Matrix([
    #     [1, 0, 0, 0],           # t-leg (index 0)
    #     [0, 1, 0, 0],         # r-leg (index 1)
    #     [0, 0, r, 0],                    # theta-leg (index 2)
    #     [0, 0, 0, r*sp.sin(theta)]       # phi-leg (index 3)
    # ]) 
    
    
    print("\n=== 1. NON-ROTATED TETRAD MATRIX (Diagonal / Spherical) ===")
    print("Rows: [t, r, theta, phi]_hat | Cols: [dt, dr, dtheta, dphi]")
    sp.pprint(Tetrad)

    

    # # --- 2. Automated Tensor Engines ---
    # # Normal vector n_mu = (-alpha, 0, 0, 0)
    # n_cov = sp.Matrix([-alpha_func, 0, 0, 0])
    # coords = [t, r, theta, phi]

    def get_dynamical_fields(alpha_idx):
        A_mu = Tetrad[alpha_idx, :]
        
        # A. Field Strength F_uv = d_u A_v - d_v A_u
        F_cov = sp.zeros(4, 4)
        for u in range(4):
            for v in range(4):
                F_cov[u, v] = sp.simplify(sp.diff(A_mu[v], coords[u]) - sp.diff(A_mu[u], coords[v]))
        
        # Raise Indices: F^uv = g^ua g^vb F_ab
        F_up = sp.simplify(g_inv * F_cov * g_inv.T)
        
        # B. Electric Field Vector: E^u = n_v * F^uv
        E_vec = [sp.simplify(sum(n_cov[v] * F_up[u, v] for v in range(4))) for u in range(4)]
        
        # C. Hodge Dual: (*F)^uv = (1/2) * (1/sqrt(-g)) * epsilon^uvsr * F_sr
        dual_F = sp.zeros(4, 4)
        for u in range(4):
            for v in range(4):
                if u == v: continue
                hodge_val = 0
                for s in range(4):
                    for r_idx in range(4):
                        eps = sp.LeviCivita(u, v, s, r_idx)
                        if eps != 0:
                            hodge_val += -eps * F_cov[s, r_idx]
                dual_F[u, v] = sp.simplify(hodge_val / (2 * sqrt_det_g))
        
        # D. Magnetic Field Vector: B^u = n_v * (*F)^uv
        B_vec = [sp.simplify(sum(n_cov[v] * dual_F[u, v] for v in range(4))) for u in range(4)]
        
        return E_vec, B_vec
    
    def constitutive_relations(E_Results, B_Results, A_mu):
        E_hat = sp.zeros(4, 4)
        B_hat = sp.zeros(4, 4)
        
        for a in range(4):
            for i_hat in range(4):
                E_hat[a, i_hat] = sp.simplify(sum(E_Results[a][mu] * A_mu[i_hat, mu] for mu in range(4)))
                B_hat[a, i_hat] = sp.simplify(sum(B_Results[a][mu] * A_mu[i_hat, mu] for mu in range(4)))

        D_hat = sp.zeros(4, 4)
        H_hat = sp.zeros(4, 4)
        
        tr_E = sum(E_hat[l, l] for l in range(1, 4))
        tr_B = sum(B_hat[l, l] for l in range(1, 4))

        # --- Time-Leg Projections (hat_alpha = 0) ---
        for i in range(1, 4):
            eps_sum_D = sum(sum(sp.LeviCivita(i, j, k) * B_hat[j, k] for k in range(1, 4)) for j in range(1, 4))
            D_hat[0, i] = sp.simplify(-eps_sum_D)
            
            eps_sum_H = sum(sum(sp.LeviCivita(i, j, k) * E_hat[j, k] for k in range(1, 4)) for j in range(1, 4))
            H_hat[0, i] = sp.simplify(half * eps_sum_H)

        # --- Spatial-Leg Projections (hat_alpha = 1, 2, 3) ---
        for k in range(1, 4):
            for i in range(1, 4):
                delta_ki = 1 if k == i else 0
                
                D_hat[k, i] = sp.simplify(-half * (E_hat[k, i] + E_hat[i, k]) + delta_ki * tr_E)
                
                eps_term = sum(sp.LeviCivita(k, i, j) * E_hat[0, j] for j in range(1, 4))
                H_hat[k, i] = sp.simplify(-B_hat[i, k] + half * delta_ki * tr_B + eps_term)

        return E_hat, B_hat, D_hat, H_hat
    
    def calculate_charges_and_currents(E_hat, B_hat, D_hat, H_hat):
        rho = sp.zeros(4, 1) 
        s = sp.zeros(4, 4)   
        
        def contract_alpha(V, W, col_V, col_W):
            return sum(V[alpha, col_V] * W[alpha, col_W] for alpha in range(4))

        ED_scalar = sp.simplify(sum(contract_alpha(E_hat, D_hat, k, k) for k in range(1, 4)))
        BH_scalar = sp.simplify(sum(contract_alpha(B_hat, H_hat, k, k) for k in range(1, 4)))
        
        # --- Eq 134: rho_0 (Gravitational Energy Density) ---
        rho[0] = sp.simplify(-half * (ED_scalar + BH_scalar))
        
        # --- Eq 135: rho_i (Gravitational Momentum Density) ---
        for i in range(1, 4):
            val = 0
            for j in range(1, 4):
                for k in range(1, 4):
                    eps = sp.LeviCivita(i, j, k)
                    if eps != 0:
                        val += eps * contract_alpha(B_hat, D_hat, j, k)
            rho[i] = sp.simplify(-val)
            
        # --- Eq 136: s^i_0 (Gravitational Energy Flux / Poynting Vector) ---
        for i in range(1, 4):
            val = 0
            for j in range(1, 4):
                for k in range(1, 4):
                    eps = sp.LeviCivita(i, j, k)
                    if eps != 0:
                        val += eps * contract_alpha(E_hat, H_hat, j, k)
            s[0, i] = sp.simplify(-val)
            
        # --- Eq 137: s^i_j (Gravitational Momentum Fluxes) ---
        for j in range(1, 4):
            for i in range(1, 4):
                delta_ij = 1 if i == j else 0
                term1 = contract_alpha(E_hat, D_hat, j, i)
                term2 = contract_alpha(B_hat, H_hat, i, j)
                term3 = -half * delta_ij * (ED_scalar + BH_scalar)
                s[j, i] = sp.simplify(term1 + term2 + term3)

        print("\n=== PRL 2025 SPACETIME CHARGES & CURRENTS Densities (Eqs 25-28) ===")
        print("rho_hat (Charge densities): [rho_0, rho_r, rho_theta, rho_phi]^T")
        sp.pprint(rho)
        print("\nj_hat (Current densities): Rows = hat_alpha (0, r, theta, phi) | Cols = hat_i (dt, dr, dtheta, dphi) directions")
        sp.pprint(s)
        
        
        return rho, s
    
    def calculate_PRL_charges_and_currents(E_hat, B_hat, D_hat, H_hat, sqrt_det_g, alpha_func):
        sqrt_gamma = sp.simplify(sqrt_det_g / alpha_func)
        
        q = sp.zeros(4, 1) 
        j = sp.zeros(4, 4) 
        
        def contract_alpha(V, W, col_V, col_W):
            return sum(V[alpha, col_V] * W[alpha, col_W] for alpha in range(4))

        ED_scalar = sp.simplify(sum(contract_alpha(E_hat, D_hat, k, k) for k in range(1, 4)))
        BH_scalar = sp.simplify(sum(contract_alpha(B_hat, H_hat, k, k) for k in range(1, 4)))
        
        q[0] = sp.simplify(-half * sqrt_gamma * (ED_scalar + BH_scalar))
        
        for i in range(1, 4):
            val = 0
            for k in range(1, 4):
                for l in range(1, 4):
                    eps = sp.LeviCivita(i, k, l)
                    if eps != 0:
                        val += eps * contract_alpha(B_hat, D_hat, k, l)
            q[i] = sp.simplify(-sqrt_gamma * val)
            
        for i in range(1, 4):
            val = 0
            for k in range(1, 4):
                for l in range(1, 4):
                    eps = sp.LeviCivita(i, k, l)
                    if eps != 0:
                        val += eps * contract_alpha(E_hat, H_hat, k, l)
            j[0, i] = sp.simplify(-sqrt_gamma * val)
            
        for leg in range(1, 4): 
            for comp in range(1, 4):
                delta_leg_comp = 1 if comp == leg else 0
                term1 = contract_alpha(E_hat, D_hat, leg, comp)
                term2 = contract_alpha(B_hat, H_hat, comp, leg)
                term3 = -half * delta_leg_comp * (ED_scalar + BH_scalar)
                j[leg, comp] = sp.simplify(sqrt_gamma * (term1 + term2 + term3))

        print("\n=== PRL 2025 SPACETIME CHARGES & CURRENTS (Eqs 25-28) ===")
        print("q_hat (Charges): [q_0, q_r, q_theta, q_phi]^T")
        sp.pprint(q)
        print("\nj_hat (Currents): Rows = hat_alpha (0, r, theta, phi) | Cols = hat_i (dt, dr, dtheta, dphi) directions")
        sp.pprint(j)
        
        return q, j
    
    def plot_analytical_fields(E_hat, q_hat, rho_hat, M_val=1.0):
        print("\n=== 4. GENERATING ANALYTICAL PLOTS (CARTESIAN) ===")
        # 1. Setup 2D Cartesian Grid (X-Z plane, y=0)
        grid_lim = 8.0
        resolution = 250
        x_vals = np.linspace(-grid_lim, grid_lim, resolution)
        z_vals = np.linspace(-grid_lim, grid_lim, resolution)
        X, Z = np.meshgrid(x_vals, z_vals)
        Y = np.zeros_like(X) # Evaluate exactly on the equatorial slice y=0
        
        R = np.sqrt(X**2 + Z**2)
        
        # Mask out the interior of the black hole 
        # In isotropic coords, the horizon is at r_bar = M/2
        horizon_r_bar = M_val / 2.0
        mask = R > (1.05 * horizon_r_bar)
        
        # Safe arrays bypass math domain errors inside the horizon
        X_safe = np.where(mask, X, 1.05 * horizon_r_bar)
        Y_safe = np.where(mask, Y, 0.0)
        Z_safe = np.where(mask, Z, 0.0)

        # --- 3. Lambdify SymPy Expressions ---
        print("Lambdifying exact symbolic expressions into NumPy functions...")
        
        # Extract Ex (index 1) and Ez (index 3) from the time-leg (index 0) of E_hat
        Ex_sym = sp.simplify(E_hat[0, 1].subs(M, M_val))
        Ez_sym = sp.simplify(E_hat[0, 3].subs(M, M_val))
        q0_sym = sp.simplify(q_hat[0].subs(M, M_val))
        rho0_sym = sp.simplify(rho_hat[0].subs(M, M_val))
        
        # Compile to NumPy using the new (x, y, z) symbols
        Ex_func = lambdify((x, y, z), Ex_sym, "numpy")
        Ez_func = lambdify((x, y, z), Ez_sym, "numpy")
        q0_func = lambdify((x, y, z), q0_sym, "numpy")
        rho0_func = lambdify((x, y, z), rho0_sym, "numpy")
        
        # --- 4. Evaluate on Grid ---
        print("Evaluating analytical fields over the 2D grid...")
        Ex_num = np.zeros_like(R)
        Ez_num = np.zeros_like(R)
        q0_num = np.zeros_like(R)
        rho0_num = np.zeros_like(R)
        
        # Vectorized evaluation directly using X, Y, Z
        Ex_num[mask] = Ex_func(X_safe[mask], Y_safe[mask], Z_safe[mask])
        Ez_num[mask] = Ez_func(X_safe[mask], Y_safe[mask], Z_safe[mask])
        q0_num[mask] = q0_func(X_safe[mask], Y_safe[mask], Z_safe[mask])
        rho0_num[mask] = rho0_func(X_safe[mask], Y_safe[mask], Z_safe[mask])
        
        # Apply visual mask
        Ex_num[~mask] = np.nan
        Ez_num[~mask] = np.nan
        q0_num[~mask] = np.nan
        rho0_num[~mask] = np.nan

        # --- 5. Rendering ---
        print("Rendering Plot...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Panel 1: Physical Density
        ax1 = axes[0]
        rho_max = np.nanmax(np.abs(rho0_num))
        if np.isnan(rho_max) or rho_max == 0: rho_max = 1.0
        
        cmap1 = ax1.pcolormesh(X, Z, rho0_num, shading='auto', cmap='RdBu_r', vmin=-rho_max, vmax=rho_max)
        fig.colorbar(cmap1, ax=ax1, label=r"Gravitational Energy Density ($\rho_{\hat{0}}$)")
        ax1.streamplot(x_vals, z_vals, Ex_num, Ez_num, color='black', density=1.5, linewidth=0.8)
        ax1.add_patch(plt.Circle((0, 0), horizon_r_bar, color='black', zorder=10))
        ax1.set_aspect('equal')
        ax1.set_title(r"True Energy Density: $\rho_{\hat{0}}$ (Cartesian Isotropic)", fontsize=14, fontweight='bold')
        ax1.set_xlabel("X (M)", fontsize=12)
        ax1.set_ylabel("Z (M)", fontsize=12)

        # Panel 2: Macroscopic Charge
        ax2 = axes[1]
        q_max = np.nanmax(np.abs(q0_num))
        if np.isnan(q_max) or q_max == 0: q_max = 1.0 
        
        cmap2 = ax2.pcolormesh(X, Z, q0_num, shading='auto', cmap='RdBu_r', vmin=-q_max, vmax=q_max)
        fig.colorbar(cmap2, ax=ax2, label=r"Spacetime Energy Charge ($q_{\hat{0}}$)")
        ax2.streamplot(x_vals, z_vals, Ex_num, Ez_num, color='black', density=1.5, linewidth=0.8)
        ax2.add_patch(plt.Circle((0, 0), horizon_r_bar, color='black', zorder=10))
        ax2.set_aspect('equal')
        ax2.set_title(r"Macroscopic Charge: $q_{\hat{0}}$", fontsize=14, fontweight='bold')
        ax2.set_xlabel("X (M)", fontsize=12)

        out_name = "Analytical_Density_vs_Charge_Cartesian.png"
        plt.tight_layout()
        plt.savefig(out_name, dpi=200)
        print(f"🎉 Plot successfully saved to {out_name}\n")

    # --- 3. Execution and Formatting ---
    labels = ['t', 'x', 'y', 'z']
    E_Results = []
    B_Results = []
    
    print("\nCalculating Automated Electric and Magnetic Fields...")
    for i in range(4):
        E, B = get_dynamical_fields(i)
        E_Results.append(E)
        B_Results.append(B)

    print("\n--- Electric Field Components E^a_i ---")
    # Just printing the first spatial leg to keep the terminal output clean
    for i in range(4):
        print(f"E^({labels[i]}x) = ")
        sp.pprint(sp.simplify(E_Results[i][1]))
        print("")

    B_Matrix_Rows = []
    for i in range(4):
        B_Matrix_Rows.append([B_Results[i][1], B_Results[i][2], B_Results[i][3]])
    B_Matrix = sp.Matrix(B_Matrix_Rows)

    print("\n=== 3. MAGNETIC FIELD MATRIX (Automated Hodge Dual) ===")
    sp.pprint(B_Matrix)

    E_hat, B_hat, D_hat, H_hat = constitutive_relations(E_Results, B_Results, Tetrad)

    # Calculate Charges
    rho_hat, s_hat = calculate_charges_and_currents(E_hat, B_hat, D_hat, H_hat)
    q_hat, j_hat = calculate_PRL_charges_and_currents(E_hat, B_hat, D_hat, H_hat, sqrt_det_g, alpha_func)

    # Trigger Visualization
    plot_analytical_fields(E_hat, q_hat, rho_hat, M_val=1.0)

if __name__ == "__main__":
    calculate_automated_fields()