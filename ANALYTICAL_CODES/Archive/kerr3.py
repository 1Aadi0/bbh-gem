import sympy as sp

def calculate_kerr_fields_exact():
    # ==========================================
    # 1. SETUP & PARAMETERS
    # ==========================================
    print("Initializing Kerr Spacetime & Calculation Pipeline...")
    
    # Coordinates
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    
    # Physical Parameters
    M = sp.symbols('M', real=True, positive=True)
    chi = sp.symbols('chi', real=True) # Dimensionless spin
    
    # Relation: a = chi * M
    a = chi * M
    
    # Display variable for Far Field
    R = sp.symbols('R', real=True, positive=True)

    # ==========================================
    # 2. KERR METRIC (Boyer-Lindquist)
    # ==========================================
    rho2 = r**2 + a**2 * sp.cos(theta)**2
    rho = sp.sqrt(rho2)
    Delta = r**2 - 2*M*r + a**2
    Sigma = sp.sqrt((r**2 + a**2)**2 - Delta * a**2 * sp.sin(theta)**2)
    
    # Covariant Metric g_mu_nu
    # diag(-1 + 2Mr/rho2, rho2/Delta, rho2, sin2th...) + off-diagonal gtphi
    # We define it explicitly to ensure correct inversion
    g_tt = -(1 - 2*M*r/rho2)
    g_rr = rho2 / Delta
    g_thth = rho2
    g_phph = (r**2 + a**2 + 2*M*r*a**2*sp.sin(theta)**2/rho2)*sp.sin(theta)**2
    g_tph = -2*M*r*a*sp.sin(theta)**2/rho2
    
    g_cov = sp.Matrix([
        [g_tt,  0,     0,      g_tph],
        [0,     g_rr,  0,      0    ],
        [0,     0,     g_thth, 0    ],
        [g_tph, 0,     0,      g_phph]
    ])
    
    # Inverse Metric g^mu_nu (Crucial for raising indices and mixing E/B fields)
    g_inv = g_cov.inv()
    
    # Lapse Function alpha = 1 / sqrt(-g^tt)
    # Used for the normal vector n_mu = (-alpha, 0, 0, 0)
    alpha = 1 / sp.sqrt(-g_inv[0,0])

    # ==========================================
    # 3. TETRAD (Maluf Eq 55 - Static Frame)
    # ==========================================
    # Auxiliary functions for Maluf Tetrad
    psi2 = Delta - a**2 * sp.sin(theta)**2
    psi = sp.sqrt(psi2)
    chi_maluf = 2 * a * M * r # Eq 53 in Maluf
    Upsilon = sp.sqrt(psi2 * Sigma**2 + chi_maluf**2 * sp.sin(theta)**2) # Approx per Maluf defs, but let's use Eq 57 directly
    # Re-check Maluf Eq 57 definition context. Upsilon is defined.
    # Let's use the coefficients directly.
    
    A_c = psi / rho
    B_c = (chi_maluf * sp.sin(theta)**2) / (rho * psi)
    C_c = rho / sp.sqrt(Delta)
    D_c = Upsilon / (rho * psi)
    
    # Covariant Tetrad e^a_mu
    # Rows: a=0(t), 1(x), 2(y), 3(z)
    e_cov = sp.Matrix([
        [-A_c, 0, 0, -B_c],
        [0, C_c * sp.sin(theta) * sp.cos(phi), rho * sp.cos(theta) * sp.cos(phi), -D_c * sp.sin(theta) * sp.sin(phi)],
        [0, C_c * sp.sin(theta) * sp.sin(phi), rho * sp.cos(theta) * sp.sin(phi), D_c * sp.sin(theta) * sp.cos(phi)],
        [0, C_c * sp.cos(theta), -rho * sp.sin(theta), 0]
    ])

    print(e_cov)

    # ==========================================
    # 4. FIELD CALCULATION ROUTINE
    # ==========================================
    coords = [t, r, theta, phi]
    
    # Helper to clean up trig expressions to match paper format
    def simplify_trig_product(expr):
        if expr == 0: return 0
        # Expand cos(a-b) -> cos(a)cos(b) + sin(a)sin(b)
        expanded = sp.expand_trig(expr)
        # Simplify remaining terms
        return sp.simplify(expanded)

    # Generic function to get Raised Field Strength F^uv
    def get_F_raised(tetrad_idx):
        # 1. Compute Covariant F_mn = d_m A_n - d_n A_m
        A_mu = e_cov[tetrad_idx, :]
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

    # ==========================================
    # 5. CALCULATE E & B
    # ==========================================
    # Electric Field E^mu = n_lam F^mu_lam
    # Normal vector n_lam = (-alpha, 0, 0, 0)
    # E^mu = -alpha * F^mu0 = alpha * F^0mu
    
    print("\nCalculating Electric Fields...")
    E_results = []
    for i in range(4): # For each tetrad leg
        F_up = get_F_raised(i)
        # E^R component corresponds to index 1 (r)
        # E^a_R = alpha * F^a_0r (where 0,r are spacetime indices)
        # Wait, F_up is F^mu^nu. E^r = alpha * F^01
        E_r = - alpha * F_up[0, 1]
        
        # Series Expand
        # Need terms up to order 1/r^4.
        # The leading term for x,y is O(a * M^2 / r^4).
        # Expand 'a' to order 2 (to keep linear 'a'), 'r' to order 5.
        term = sp.series(E_r, a, 0, 2).removeO()
        term = sp.series(term, r, sp.oo, 5).removeO()
        
        # For t-leg (i=0), we need higher order in 'a' for the quadrupole correction
        if i == 0:
            term = sp.series(E_r, a, 0, 4).removeO()
            term = sp.series(term, r, sp.oo, 5).removeO()
            
        E_results.append(simplify_trig_product(term.subs(r, R)))

    # Magnetic Field B^mu = n_lam (*F)^mu_lam
    # *F^uv = 1/2 eps^uvab F_ab
    # But simpler: B^i = epsilon^ijk (spatial) * ...
    # Let's use the paper's implied definition B^i (spatial).
    # In the local frame of n, B^i = 1/2 eps^ijk F_jk (spatial indices).
    # We need determinant of metric sqrt(-g) = alpha * sqrt(gamma) = alpha * rho^2 sin(theta) approx?
    # Exact sqrt(-g) = rho^2 sin(theta) for BL coordinates.
    det_g = rho**2 * sp.sin(theta)
    
    print("Calculating Magnetic Fields...")
    B_matrix_rows = []
    for i in range(4): # For each tetrad leg
        # Covariant F_mn is sufficient for B (spatial components)
        # B^i = (1/sqrt(-g)) * [cycle derivatives] ? No, B^i is a vector.
        # Standard definition: B^1 = (F_23)/sqrt(gamma), etc.
        # B^r = F_th_ph / sqrt(gamma)
        
        # Note: We need B^mu (contravariant spacetime index).
        # B^r = (F_23)/sqrt(det_spatial) ?
        # Let's use F_lower directly: B^1 = F_23 / (rho^2 sin theta / alpha) ??
        # Actually, let's use the robust definition: B^mu = -1/2 n_rho eps^rho mu nu sigma F_nu sigma
        # Since n = (-alpha, 0, 0, 0), B^i = -1/2 (-alpha) eps^0 i j k F_jk
        # B^i = alpha/2 * (1/sqrt(-g)) * [permutation] * F_jk
        # B^1 = alpha / (2 rho^2 sin th) * 2 * F_23 = alpha * F_23 / (rho^2 sin th)
        
        A_mu = e_cov[i, :]
        F_23 = sp.diff(A_mu[3], theta) - sp.diff(A_mu[2], phi)
        F_31 = sp.diff(A_mu[1], phi) - sp.diff(A_mu[3], r)
        F_12 = sp.diff(A_mu[2], r) - sp.diff(A_mu[1], theta)
        
        factor = alpha / (rho**2 * sp.sin(theta))
        
        B_vec = [factor * F_23, factor * F_31, factor * F_12]
        
        row_clean = []
        for j, comp in enumerate(B_vec):
            # Expansion
            
            term = sp.series(comp, a, 0, 3).removeO()
            term = sp.series(term, r, sp.oo, 4).removeO()

            # Z-leg phi-component dipole needs expansion to order a^1
            if i == 3 and j == 2:
                term = sp.series(comp, a, 0, 1).removeO()
                term = sp.series(term, r, sp.oo, 4).removeO()

            clean = simplify_trig_product(term.subs(r, R))
            row_clean.append(clean)
        B_matrix_rows.append(row_clean)

    # ==========================================
    # 6. OUTPUT GENERATION
    # ==========================================
    print("\n" + "="*30)
    print("      FINAL RESULTS")
    print("="*30)
    
    print("\n--- Electric Field (Radial Components) ---")
    labels = ['t', 'x', 'y', 'z']
    for i in range(3): # Show t, x, y (z is usually 0 or small)
        print(f"E^({labels[i]}R) = ")
        sp.pprint(E_results[i])
        print("")

    print("\n--- Magnetic Field Matrix ---")
    cols = ['r', 'theta', 'phi']
    for i in range(4):
        print(f"Row {labels[i]}:")
        for j in range(3):
            val = B_matrix_rows[i][j]
            if val != 0:
                print(f"  B^{cols[j]}:")
                sp.pprint(val)
            else:
                print(f"  B^{cols[j]}: 0")
        print("")



if __name__ == "__main__":
    calculate_kerr_fields_exact()


