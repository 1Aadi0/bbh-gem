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
    


    # #Test case
    # alpha_func = 1
    # # # Covariant Metric g_uv
    # g_cov = sp.diag(-1, 1, r**2, r**2 * sp.sin(theta)**2)
    # g_inv = g_cov.inv()
    # sqrt_det_g = sp.simplify(sp.sqrt(-g_cov.det()))
    
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

    # --- 2. Automated Tensor Engines ---
    # Normal vector n_mu = (-alpha, 0, 0, 0)
    n_cov = sp.Matrix([-alpha_func, 0, 0, 0])
    coords = [t, r, theta, phi]

    def get_dynamical_fields(alpha_idx):
        A_mu = Tetrad[alpha_idx, :]
        sp.pprint(A_mu)
        
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
    
    def constitutive_relations(E_Results, B_Results, A_mu):
        """
        Implements DGREM constitutive relations (Eqs 122-125) to map 
        Frame Fields (E, B) to Auxiliary Fields (D, H).
        """
        # 1. Project E and B to the Orthonormal Tetrad Basis
        # A^mu_{hat_i} are the columns of the inverse tetrad matrix
        A_inv = Tetrad.inv()
        E_hat = sp.zeros(4, 4)
        B_hat = sp.zeros(4, 4)
        
        for a in range(4):
            for i_hat in range(4):
                # V^{hat_a}_{hat_i} = V^{hat_a}_mu * e^mu_{hat_i}
                E_hat[a, i_hat] = sp.simplify(sum(E_Results[a][mu] * A_mu[i_hat, mu] for mu in range(4)))
                B_hat[a, i_hat] = sp.simplify(sum(B_Results[a][mu] * A_mu[i_hat, mu] for mu in range(4)))

        D_hat = sp.zeros(4, 4)
        H_hat = sp.zeros(4, 4)
        print("\n--- 2. Projected Fields in Orthonormal Tetrad Basis ---")
        print("E_hat (Electric in Tetrad): Rows [t, r, theta, phi] Legs | Cols: [t, r, theta, phi] Components")
        sp.pprint(E_hat)
        print("B_hat (Magnetic in Tetrad): Rows [t, r, theta, phi] Legs | Cols: [t, r, theta, phi] Components")
        sp.pprint(B_hat)
        # Pre-calculate spatial traces: Tr(V) = V^1_1 + V^2_2 + V^3_3
        tr_E = sum(E_hat[l, l] for l in range(1, 4))
        tr_B = sum(B_hat[l, l] for l in range(1, 4))
        print(f"trace of E: {tr_E}")
        print(f"trace of B: {tr_B}")

        # --- 2. Time-Leg Projections (hat_alpha = 0) ---
        for i in range(1, 4):
            # Eq 122: D^i_0 = - eps^{ijk} B_jk (antisymmetric part of spatial legs)
            eps_sum_D = sum(sum(sp.LeviCivita(i, j, k) * B_hat[j, k] for k in range(1, 4)) for j in range(1, 4))
            D_hat[0, i] = sp.simplify(-eps_sum_D)
            
            # Eq 124: H_{0 i} = 1/2 eps_{ijk} E^jk (antisymmetric part of spatial legs)
            eps_sum_H = sum(sum(sp.LeviCivita(i, j, k) * E_hat[j, k] for k in range(1, 4)) for j in range(1, 4))
            H_hat[0, i] = sp.simplify(0.5 * eps_sum_H)

        # --- 3. Spatial-Leg Projections (hat_alpha = 1, 2, 3) ---
        for k in range(1, 4): # Leg index (hat_k)
            for i in range(1, 4): # Component index (hat_i)
                delta_ki = 1 if k == i else 0
                
                # Eq 123: D^i_k = -1/2 (E^i_k + E^k_i) + delta^i_k * Tr(E)
                # This incorporates the symmetric part and the isotropic pressure
                D_hat[k, i] = sp.simplify(-0.5 * (E_hat[k, i] + E_hat[i, k]) + delta_ki * tr_E)
                
                # Eq 125: H_{k i} = -B_{i k} + 1/2 delta_{k i} Tr(B) - eps_{kij} E^0_j
                # Note the index swap on the first term (B_{i k}) per PRD Eq 125
                eps_term = sum(sp.LeviCivita(k, i, j) * E_hat[0, j] for j in range(1, 4))
                H_hat[k, i] = sp.simplify(-B_hat[i, k] + 0.5 * delta_ki * tr_B + eps_term)

        # ... end of constitutive_relations function
        return E_hat, B_hat, D_hat, H_hat
    
    def calculate_charges_and_currents(E_hat, B_hat, D_hat, H_hat):
        """
        Calculates the gravitational energy-momentum densities (rho) and 
        fluxes (s) based on PRD 2022 Equations 134-137.
        """
        rho = sp.zeros(4, 1) # Gravitational charge/energy densities
        s = sp.zeros(4, 4)   # Gravitational momentum fluxes
        
        # Helper for direct Einstein summation over hat_alpha.
        # V has alpha UP (E, B), W has alpha DOWN (D, H).
        def contract_alpha(V, W, col_V, col_W):
            # Sums V^{alpha}_{col_V} * W_{alpha}^{col_W}
            return sum(V[alpha, col_V] * W[alpha, col_W] for alpha in range(4))

        # Scalar contractions: E^{alpha k} D_{alpha k} and B^{alpha k} H_{alpha k}
        ED_scalar = sum(contract_alpha(E_hat, D_hat, k, k) for k in range(1, 4))
        BH_scalar = sum(contract_alpha(B_hat, H_hat, k, k) for k in range(1, 4))
        
        # --- Eq 134: rho_0 (Gravitational Energy Density) ---
        rho[0] = sp.simplify(-0.5 * (ED_scalar + BH_scalar))
        
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
                term3 = -0.5 * delta_ij * (ED_scalar + BH_scalar)
                s[j, i] = sp.simplify(term1 + term2 + term3)

        # Output formatting
        print("\n=== GRAVITATIONAL CHARGES & CURRENTS (Eqs 134-137) ===")
        print("rho_hat (Densities): [rho_0, rho_r, rho_theta, rho_phi]^T")
        sp.pprint(rho)
        print("\ns_hat (Fluxes): Rows = hat_alpha (0, r, theta, phi) | Cols = hat_i (dt, dr, dtheta, dphi) directions")
        sp.pprint(s)
        
        return rho, s
    
    def calculate_PRL_charges_and_currents(E_hat, B_hat, D_hat, H_hat, sqrt_det_g, alpha_func):
        """
        Calculates the spacetime charges (q) and currents (j) based on 
        PRL 2025 Equations 25-28.
        """
        # Calculate spatial volume element sqrt(gamma) = sqrt(-g) / alpha
        sqrt_gamma = sp.simplify(sqrt_det_g / alpha_func)
        
        q = sp.zeros(4, 1) # Spacetime charges
        j = sp.zeros(4, 4) # Spacetime currents
        
        # Helper for direct Einstein summation over hat_alpha.
        def contract_alpha(V, W, col_V, col_W):
            return sum(V[alpha, col_V] * W[alpha, col_W] for alpha in range(4))

        # Scalar contractions: E \cdot D and B \cdot H
        ED_scalar = sum(contract_alpha(E_hat, D_hat, k, k) for k in range(1, 4))
        BH_scalar = sum(contract_alpha(B_hat, H_hat, k, k) for k in range(1, 4))
        
        # --- Eq 25: q_0 (Spacetime Energy Charge) ---
        q[0] = sp.simplify(-0.5 * sqrt_gamma * (ED_scalar + BH_scalar))
        
        # --- Eq 26: q_i (Spacetime Momentum Charge) ---
        for i in range(1, 4):
            val = 0
            for k in range(1, 4):
                for l in range(1, 4):
                    eps = sp.LeviCivita(i, k, l)
                    if eps != 0:
                        val += eps * contract_alpha(B_hat, D_hat, k, l)
            q[i] = sp.simplify(-sqrt_gamma * val)
            
        # --- Eq 27: j_0 (Gravitational Poynting Current) ---
        for i in range(1, 4):
            val = 0
            for k in range(1, 4):
                for l in range(1, 4):
                    eps = sp.LeviCivita(i, k, l)
                    if eps != 0:
                        val += eps * contract_alpha(E_hat, H_hat, k, l)
            j[0, i] = sp.simplify(-sqrt_gamma * val)
            
        # --- Eq 28: j_i (Spacetime Momentum Currents) ---
        for leg in range(1, 4):      # hat_alpha leg index
            for comp in range(1, 4): # spatial component index
                delta_leg_comp = 1 if comp == leg else 0
                term1 = contract_alpha(E_hat, D_hat, comp, leg)
                term2 = contract_alpha(B_hat, H_hat, comp, leg)
                term3 = -0.5 * delta_leg_comp * (ED_scalar + BH_scalar)
                j[leg, comp] = sp.simplify(sqrt_gamma * (term1 + term2 + term3))

        # Output formatting
        print("\n=== PRL 2025 SPACETIME CHARGES & CURRENTS (Eqs 25-28) ===")
        print("q_hat (Charges): [q_0, q_r, q_theta, q_phi]^T")
        sp.pprint(q)
        print("\nj_hat (Currents): Rows = hat_alpha (0, r, theta, phi) | Cols = hat_i (dt, dr, dtheta, dphi) directions")
        sp.pprint(j)
        
        return q, j
    
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


    # Change this line:
    E_hat, B_hat, D_hat, H_hat = constitutive_relations(E_Results, B_Results, Tetrad)

    print("\n=== DGREM FIELD RESULTS (Projected Orthonormal) ===")
    print("D_hat (Auxiliary Electric): Rows [t, r, theta, phi] Legs")
    sp.pprint(D_hat)
    print("\nH_hat (Auxiliary Magnetic): Rows [t, r, theta, phi] Legs")
    sp.pprint(H_hat)

    rho_hat, s_hat = calculate_charges_and_currents(E_hat, B_hat, D_hat, H_hat)
    # Change this line:
    q_hat, j_hat = calculate_PRL_charges_and_currents(E_hat, B_hat, D_hat, H_hat, sqrt_det_g, alpha_func)

    # # Sanity Check for Vacuum Charge q0
    # # q0 = div(D_0) - (rho + kappa*P)
    # div_D0 = sp.simplify(sp.diff(sqrt_det_g * D_hat[0, 1], r) / sqrt_det_g) # Radial div for spherical
    # print("\n--- Vacuum Spacetime Charge Density (q0) ---")
    # sp.pprint(div_D0)

# if __name__ == "__main__":
#     calculate_automated_fields()

if __name__ == "__main__":
    calculate_automated_fields()