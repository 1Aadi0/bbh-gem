import sympy as sp

def calculate_automated_fields():
    # --- 1. Variables and Metric Setup ---
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    M = sp.symbols('M', real=True, positive=True)
    R_far = sp.symbols('R', real=True, positive=True) # For display in Far Field
    rs = 2 * M
    
    alpha_func = sp.sqrt(1 - rs/r)
    
    # Covariant Metric g_uv
    g_cov = sp.diag(-(alpha_func**2), 1/(alpha_func**2), r**2, r**2 * sp.sin(theta)**2)
    g_inv = g_cov.inv()
    sqrt_det_g = sp.simplify(sp.sqrt(-g_cov.det()))
    
    # Non-Rotated Tetrad Legs A_mu^(hat_a)
    Tetrad = sp.Matrix([
        [alpha_func, 0, 0, 0],           # t-leg (index 0)
        [0, 1/alpha_func, 0, 0],         # r-leg (index 1)
        [0, 0, r, 0],                    # theta-leg (index 2)
        [0, 0, 0, r*sp.sin(theta)]       # phi-leg (index 3)
    ])
    
    print("\n=== 1. NON-ROTATED TETRAD MATRIX (Diagonal / Spherical) ===")
    print("Rows: [t, r, theta, phi]_hat | Cols: [dt, dr, dtheta, dphi]")
    sp.pprint(Tetrad)

    # --- 2. Automated Tensor Engines ---
    # Normal vector n_mu = (-alpha, 0, 0, 0)
    n_cov = sp.Matrix([-alpha_func, 0, 0, 0])
    coords = [t, r, theta, phi]

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
        # Dynamically contracts covariant n with contravariant F
        E_vec = [sp.simplify(sum(n_cov[v] * F_up[u, v] for v in range(4))) for u in range(4)]
        
        # C. Hodge Dual: (*F)^uv = (1/2) * (1/sqrt(-g)) * epsilon^uvsr * F_sr
        # Note: We use the 4D Levi-Civita symbol and divide by sqrt(-g) to get the tensor
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

    # --- 3. Execution and Formatting ---
    labels = ['t', 'r', 'theta', 'phi']
    E_Results = []
    B_Results = []

    print("\nCalculating Automated Electric and Magnetic Fields...")
    for i in range(4):
        E, B = get_dynamical_fields(i)
        E_Results.append(E)
        B_Results.append(B)

    print("\n--- Electric Field (Radial Components E^a_R) ---")
    for i in range(4):
        print(f"E^({labels[i]}R) = ")
        # Substitute r -> R for far-field display as in original code
        sp.pprint(sp.simplify(E_Results[i][1].subs(r, R_far)))
        print("")

    # Construct Magnetic Matrix as in Original Output
    # Columns map to [B^r, B^theta, B^phi] (indices 1, 2, 3)
    B_Matrix_Rows = []
    for i in range(4):
        B_Matrix_Rows.append([B_Results[i][1], B_Results[i][2], B_Results[i][3]])
    
    B_Matrix = sp.Matrix(B_Matrix_Rows)

    print("\n=== 3. MAGNETIC FIELD MATRIX (Automated Hodge Dual) ===")
    print("Rows: Tetrad Legs [t, r, theta, phi]")
    print("Cols: Field Components [B^r, B^theta, B^phi]")
    sp.pprint(B_Matrix)

    print("\n--- Verification of B^z (Dipole Term) ---")
    print("Extracting B^phi component of the phi-leg (Row 4, Col 3)...")
    B_z_phi = B_Matrix[3, 2]
    
    print("\nExact Expression for B_phi component:")
    sp.pprint(B_z_phi)

    print("\nFar Field Approximation (Series expand around M=0):")
    # Result for non-rotated spherical will be 0; rotated will be M/r^3
    B_z_phi_approx = B_z_phi.series(M, 0, 2).removeO()
    sp.pprint(B_z_phi_approx)

if __name__ == "__main__":
    calculate_automated_fields()