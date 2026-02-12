import sympy as sp

def calculate_kerr_fields_exact():
    print("1. Initializing Symbols...")
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    M = sp.symbols('M', real=True, positive=True)
    chi = sp.symbols('chi', real=True)
    a = chi * M
    
    # Define missing variables to prevent NameError
    rho2 = r**2 + a**2 * sp.cos(theta)**2
    rho = sp.sqrt(rho2)
    Delta = r**2 - 2*M*r + a**2
    psi = sp.sqrt(Delta - a**2 * sp.sin(theta)**2) # Example Maluf psi
    Upsilon = sp.symbols('Upsilon') # Placeholder so the matrix can form
    chi_maluf = 2 * a * M * r

    print("2. Building Tetrad...")
    A_c, B_c, C_c, D_c = sp.symbols('A_c B_c C_c D_c') # Placeholders
    
    # This is where your code was likely crashing
    e_cov = sp.Matrix([
        [-A_c, 0, 0, -B_c],
        [0, C_c * sp.sin(theta) * sp.cos(phi), rho * sp.cos(theta) * sp.cos(phi), -D_c * sp.sin(theta) * sp.sin(phi)],
        [0, C_c * sp.sin(theta) * sp.sin(phi), rho * sp.cos(theta) * sp.sin(phi), D_c * sp.sin(theta) * sp.cos(phi)],
        [0, C_c * sp.cos(theta), -rho * sp.sin(theta), 0]
    ])

    print("3. Matrix Formed Successfully!")
    print(e_cov)

if __name__ == "__main__":
    calculate_kerr_fields_exact()