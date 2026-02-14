import sympy as sp

def calculate_schwarzschild_fields():
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    M = sp.symbols('M', real=True, positive=True)
    rs = 2 * M
    alpha = sp.sqrt(1 - rs/r)
    sqrt_gamma = (r**2 * sp.sin(theta)) / alpha
    # Display variable for Far Field
    R = sp.symbols('R', real=True, positive=True)
    Tetrad_Spherical = sp.Matrix([
        [alpha, 0, 0, 0],           
        [0, 1/alpha, 0, 0],         
        [0, 0, r, 0],               
        [0, 0, 0, r*sp.sin(theta)]  
    ])
    
    print("\n=== 1. NON-ROTATED TETRAD MATRIX (Diagonal / Spherical) ===")
    print("Rows: [t, r, theta, phi]_hat | Cols: [dt, dr, dtheta, dphi]")
    sp.pprint(Tetrad_Spherical)

    def simplify_trig_product(expr):
        if expr == 0: return 0
        # Expand cos(a-b) -> cos(a)cos(b) + sin(a)sin(b)
        expanded = sp.expand_trig(expr)
        # Simplify remaining terms
        return sp.simplify(expanded)
    
    # M_sph_to_cart_basis = sp.Matrix([
    #     [sp.sin(theta)*sp.cos(phi), sp.sin(theta)*sp.sin(phi), sp.cos(theta)], 
    #     [sp.cos(theta)*sp.cos(phi), sp.cos(theta)*sp.sin(phi), -sp.sin(theta)], 
    #     [-sp.sin(phi),              sp.cos(phi),               0]              
    # ])
    
    
    # R_spatial = M_sph_to_cart_basis.T
    
    # print("\n--- Deriving Rotation Matrix ---")
    # print("1. Defined Spherical basis vectors in Cartesian components.")
    # print("2. Constructed Basis Transformation Matrix M.")
    # print("3. Calculated Rotation Matrix R = M.T (Transpose/Inverse).")
    # sp.pprint(R_spatial)
    
    
    # E_spatial_spherical = Tetrad_Spherical[1:4, :]
    # E_spatial_cartesian = R_spatial * E_spatial_spherical
    
    
    # Tetrad_Rotated = sp.Matrix.vstack(Tetrad_Spherical[0, :], E_spatial_cartesian)
    
    # print("\n=== 2. ROTATED TETRAD MATRIX (Cartesian Aligned) ===")
    # print("Rows: [t, x, y, z]_hat | Cols: [dt, dr, dtheta, dphi]")
    # # sp.pprint(sp.simplify(Tetrad_Rotated))
    g_tt = -(1 - rs/r)
    g_rr = 1/(1 - rs/r)
    g_thth = r**2
    g_phph = (r**2)*sp.sin(theta)**2
    
    g_cov = sp.Matrix([
        [g_tt,  0,     0,      0],
        [0,     g_rr,  0,      0    ],
        [0,     0,     g_thth, 0    ],
        [0, 0,     0,      g_phph]
    ])
    
    # Inverse Metric g^mu_nu (Crucial for raising indices and mixing E/B fields)
    g_inv = g_cov.inv()
    
    # Lapse Function alpha = 1 / sqrt(-g^tt)
    # Used for the normal vector n_mu = (-alpha, 0, 0, 0)
    alpha = 1 / sp.sqrt(-g_inv[0,0])
    
    coords = [t, r, theta, phi]
    def get_F_raised(tetrad_idx):
        # 1. Compute Covariant F_mn = d_m A_n - d_n A_m
        A_mu = Tetrad_Spherical[tetrad_idx, :]
        F_lower = sp.zeros(4, 4)
        for mu in range(4):
            for nu in range(4):
                if mu >= nu: continue # Antisymmetric
                val = sp.diff(A_mu[nu], coords[mu]) - sp.diff(A_mu[mu], coords[nu])
                F_lower[mu, nu] = val
                F_lower[nu, mu] = -val
        
        # 2. Raise Indices: F^mn = g^ma g^nb F_ab
        # We use matrix multiplication F_up = G_inv * F_down * G_inv.T
        F_upper = g_inv * F_lower * g_inv.T
        return F_upper
    print("\nCalculating Electric Fields...")
    E_results = []
    for i in range(4): # For each tetrad leg
        F_up = get_F_raised(i)
        # E^R component corresponds to index 1 (r)
        # E^a_R = alpha * F^a_0r (where 0,r are spacetime indices)
        # Wait, F_up is F^mu^nu. E^r = alpha * F^01
        E_r = - alpha * F_up[0, 1]
        
        # # Series Expand
        # # Need terms up to order 1/r^4.
        # # The leading term for x,y is O(a * M^2 / r^4).
        # # Expand 'a' to order 2 (to keep linear 'a'), 'r' to order 5.
        # term = sp.series(E_r, a, 0, 2).removeO()
        # term = sp.series(term, r, sp.oo, 5).removeO()
        
        # # For t-leg (i=0), we need higher order in 'a' for the quadrupole correction
        # if i == 0:
        #     term = sp.series(E_r, a, 0, 4).removeO()
        #     term = sp.series(term, r, sp.oo, 5).removeO()
            
        E_results.append(simplify_trig_product(E_r.subs(r, R)))
    print("\n--- Electric Field (Radial Components) ---")
    labels = ['t', 'x', 'y', 'z']
    for i in range(3): # Show t, x, y (z is usually 0 or small)
        print(f"E^({labels[i]}R) = ")
        sp.pprint(E_results[i])
        print("")
    
    def levi_civita(i, j, k):
        if (i, j, k) in [(1, 2, 3), (2, 3, 1), (3, 1, 2)]: return 1
        if (i, j, k) in [(3, 2, 1), (1, 3, 2), (2, 1, 3)]: return -1
        return 0

    B_Matrix_Rows = []
    
    for alpha_idx in range(4):
        
        A_vec = Tetrad_Spherical[alpha_idx, :]
        B_vec_components = [0, 0, 0] 
        
        
        for i_comp in range(1, 4): 
            curl_sum = 0
            for j in range(1, 4):
                for k in range(1, 4):
                    epsilon = levi_civita(i_comp, j, k)
                    if epsilon != 0:
                        # F_jk = d_j A_k - d_k A_j
                        d_j_Ak = sp.diff(A_vec[k], coords[j])
                        d_k_Aj = sp.diff(A_vec[j], coords[k])
                        curl_sum += epsilon * (d_j_Ak-d_k_Aj)
            
            B_val = sp.simplify((curl_sum/ (2*sqrt_gamma)))
            B_vec_components[i_comp-1] = B_val
            
        B_Matrix_Rows.append(B_vec_components)

    B_Matrix = sp.Matrix(B_Matrix_Rows)
    
    print("\n=== 3. MAGNETIC FIELD MATRIX (Rotated Frame) ===")
    print("Rows: Tetrad Legs [t, x, y, z]")
    print("Cols: Field Components [B^r, B^theta, B^phi]")
    sp.pprint(B_Matrix)
    
    print("\n--- Verification of B^z (Dipole Term) ---")
    print("Extracting B^z_phi (Row 4, Col 3)...")
    
    B_z_phi = B_Matrix[3, 2]
    
    print("\nExact Expression for B^z_phi:")
    sp.pprint(B_z_phi)
    
    print("\nFar Field Approximation (Series expand around M=0):")
    # series(M, 0, 2) expands up to order M^1
    B_z_phi_approx = B_z_phi.series(M, 0, 2).removeO()
    sp.pprint(B_z_phi_approx)
    

if __name__ == "__main__":
    calculate_schwarzschild_fields()