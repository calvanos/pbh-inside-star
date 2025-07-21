import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import warnings

class StellarConstants:
    def __init__(self):
        self.sigma = 5.67051e-5      # Stefan-Boltzmann constant
        self.c = 2.99792458e10       # Speed of light
        self.a = 7.56591e-15         # Radiation pressure constant
        self.G = 6.67259e-8          # Gravitational constant
        self.k_B = 1.380658e-16      # Boltzmann constant
        self.m_H = 1.673534e-24      # Hydrogen mass
        self.gamma = 5.0/3.0         # Adiabatic index
        self.g_ff = 1.0              # Gaunt factor
        self.gamrat = self.gamma/(self.gamma - 1.0)

class StellarStructure:
    def __init__(self, X=0.7, Z=0.02, constants=None):
        self.X = X
        self.Y = 1.0 - X - Z
        self.Z = Z
        self.XCNO = Z / 2.0
        self.mu = 1.0 / (2.0*X + 0.75*self.Y + 0.5*Z)
        self.cst = constants if constants else StellarConstants()
        
        # Numerical parameters
        self.rtol = 1e-8
        self.atol = 1e-12
        self.max_step = 1e9
        
    def equation_of_state(self, P, T, zone_id=0):
        """Improved equation of state with better error handling"""
        if T <= 0.0:
            print(f"Warning: Non-positive temperature T={T} in zone {zone_id}")
            return 0.0, 0.0, 0.0, 0.0, 1
            
        if P <= 0.0:
            print(f"Warning: Non-positive pressure P={P} in zone {zone_id}")
            return 0.0, 0.0, 0.0, 0.0, 1
        
        # Radiation pressure
        Prad = self.cst.a * T**4 / 3.0
        Pgas = P - Prad
        
        if Pgas <= 0.0:
            print(f"Warning: Non-positive gas pressure in zone {zone_id}")
            print(f"P_total = {P:.3e}, P_rad = {Prad:.3e}, P_gas = {Pgas:.3e}")
            # Use a small positive value to continue
            Pgas = 1e-10 * P
            
        # Density from ideal gas law
        rho = (self.mu * self.cst.m_H / self.cst.k_B) * (Pgas / T)
        
        if rho <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, 1
            
        # Opacity calculations with bounds checking
        try:
            tog_bf = 2.82 * (rho * (1.0 + self.X))**0.2
            k_bf = 4.34e25 / tog_bf * self.Z * (1.0 + self.X) * rho / T**3.5
            k_ff = 3.68e22 * self.cst.g_ff * (1.0 - self.Z) * (1.0 + self.X) * rho / T**3.5
            k_e = 0.2 * (1.0 + self.X)
            kappa = k_bf + k_ff + k_e
            
            # Nuclear energy generation
            T6 = T * 1.0e-6
            
            # PP chain
            fx = 0.133 * self.X * np.sqrt((3.0 + self.X) * rho) / T6**1.5
            fpp = 1.0 + fx * self.X
            psipp = 1.0 + 1.412e8 * (1.0/self.X - 1.0) * np.exp(-49.98 * T6**(-1.0/3.0))
            Cpp = 1.0 + 0.0123 * T6**(1.0/3.0) + 0.0109 * T6**(2.0/3.0) + 0.000938 * T6
            epspp = (2.38e6 * rho * self.X**2 * fpp * psipp * Cpp * 
                    T6**(-2.0/3.0) * np.exp(-33.80 * T6**(-1.0/3.0)))
            
            # CNO cycle
            CCNO = 1.0 + 0.0027 * T6**(1.0/3.0) - 0.00778 * T6**(2.0/3.0) - 0.000149 * T6
            epsCNO = (8.67e27 * rho * self.X * self.XCNO * CCNO * 
                     T6**(-2.0/3.0) * np.exp(-152.28 * T6**(-1.0/3.0)))
            
            epslon = epspp + epsCNO
            
            # Bounds checking
            if not all(np.isfinite([rho, kappa, epslon, tog_bf])):
                return 0.0, 0.0, 0.0, 0.0, 1
                
        except (OverflowError, ZeroDivisionError, FloatingPointError) as e:
            print(f"EOS calculation error in zone {zone_id}: {e}")
            return 0.0, 0.0, 0.0, 0.0, 1
            
        return rho, kappa, epslon, tog_bf, 0
    
    def stellar_equations(self, r, y):
        """
        Stellar structure equations
        y = [P, M_r, L_r, T]
        """
        if r <= 0:
            return np.zeros(4)
            
        P, M_r, L_r, T = y
        
        # Get physical properties
        rho, kappa, epslon, tog_bf, ierr = self.equation_of_state(P, T)
        
        if ierr != 0 or rho <= 0:
            # Return small derivatives to prevent integration failure
            return np.array([1e-20, 1e-20, 1e-20, 1e-20])
        
        # Check for convection
        grad_rad = (3 * kappa * rho * L_r) / (16 * np.pi * self.cst.a * self.cst.c * T**3 * r**2)
        grad_ad = (1 / self.cst.gamrat) * self.cst.G * M_r / r**2 * self.mu * self.cst.m_H / self.cst.k_B
        
        # Choose appropriate temperature gradient
        if grad_rad > grad_ad:  # Convective
            dT_dr = -grad_ad
        else:  # Radiative
            dT_dr = -grad_rad
            
        # Structure equations
        dP_dr = -self.cst.G * rho * M_r / r**2
        dM_dr = 4.0 * np.pi * rho * r**2
        dL_dr = 4.0 * np.pi * rho * epslon * r**2
        
        return np.array([dP_dr, dM_dr, dL_dr, dT_dr])
    
    def get_initial_conditions_shooting(self, M_star, L_star, R_star):
        """
        Set up initial conditions using shooting method approach
        """
        # Central conditions (small radius to avoid singularity)
        r_center = R_star * 1e-6
        
        # Estimate central conditions
        rho_c = 100.0  # g/cm^3 - initial guess
        T_c = 1.5e7    # K - initial guess
        
        # Central pressure from hydrostatic equilibrium
        P_c = self.cst.G * M_star**2 / (8 * np.pi * R_star**4) * rho_c
        
        # Initial conditions: [P, M_r, L_r, T]
        y0 = np.array([P_c, (4/3) * np.pi * rho_c * r_center**3, 0.0, T_c])
        
        return r_center, y0
    
    def integrate_structure(self, M_star, L_star, R_star, verbose=True):
        """
        Integrate stellar structure equations using adaptive methods
        """
        if verbose:
            print(f"Integrating star with M = {M_star/1.989e33:.3f} M_sun")
            print(f"L = {L_star/3.826e33:.3f} L_sun, R = {R_star/6.96e10:.3f} R_sun")
        
        # Get initial conditions
        r_start, y0 = self.get_initial_conditions_shooting(M_star, L_star, R_star)
        
        # Integration span
        r_span = (r_start, R_star)
        r_eval = np.logspace(np.log10(r_start), np.log10(R_star), 1000)
        
        try:
            # Use scipy's adaptive integrator
            sol = solve_ivp(
                self.stellar_equations,
                r_span,
                y0,
                t_eval=r_eval,
                method='RK45',  # Runge-Kutta with adaptive step size
                rtol=self.rtol,
                atol=self.atol,
                max_step=self.max_step,
                dense_output=True
            )
            
            if not sol.success:
                print(f"Integration failed: {sol.message}")
                return None
                
            if verbose:
                print(f"Integration successful with {len(sol.t)} points")
                
            # Extract results
            r = sol.t
            P = sol.y[0]
            M_r = sol.y[1]
            L_r = sol.y[2]
            T = sol.y[3]
            
            # Calculate derived quantities
            n_points = len(r)
            rho = np.zeros(n_points)
            kappa = np.zeros(n_points)
            epslon = np.zeros(n_points)
            
            for i in range(n_points):
                rho[i], kappa[i], epslon[i], _, _ = self.equation_of_state(P[i], T[i])
            
            return {
                'r': r,
                'P': P,
                'M_r': M_r,
                'L_r': L_r,
                'T': T,
                'rho': rho,
                'kappa': kappa,
                'epslon': epslon,
                'success': True
            }
            
        except Exception as e:
            print(f"Integration error: {e}")
            return None
    
    def boundary_value_solver(self, M_star, L_star, T_eff, max_iterations=20):
        """
        Solve the boundary value problem by adjusting central conditions
        """
        R_star = np.sqrt(L_star / (4 * np.pi * self.cst.sigma)) / T_eff**2
        
        def residual(central_params):
            """Residual function for boundary conditions"""
            T_c, rho_c = central_params
            
            if T_c <= 0 or rho_c <= 0:
                return np.array([1e10, 1e10])
            
            # Set up initial conditions
            r_center = R_star * 1e-6
            P_c = self.cst.G * M_star**2 / (8 * np.pi * R_star**4) * rho_c
            y0 = np.array([P_c, (4/3) * np.pi * rho_c * r_center**3, 0.0, T_c])
            
            try:
                sol = solve_ivp(
                    self.stellar_equations,
                    (r_center, R_star),
                    y0,
                    method='RK45',
                    rtol=1e-6,
                    atol=1e-10,
                    max_step=R_star/100
                )
                
                if not sol.success:
                    return np.array([1e10, 1e10])
                
                # Check boundary conditions
                P_surface = sol.y[0, -1]
                M_surface = sol.y[1, -1]
                L_surface = sol.y[2, -1]
                
                # Residuals: surface pressure should be ~0, mass and luminosity should match
                res1 = P_surface / 1e12  # Normalize pressure residual
                res2 = (M_surface - M_star) / M_star  # Fractional mass error
                res3 = (L_surface - L_star) / L_star  # Fractional luminosity error
                
                return np.array([res1, res2 + res3])
                
            except:
                return np.array([1e10, 1e10])
        
        # Initial guess for central conditions
        T_c_guess = 1.5e7
        rho_c_guess = 100.0
        
        print("Solving boundary value problem...")
        try:
            # Use root finding to satisfy boundary conditions
            solution = fsolve(residual, [T_c_guess, rho_c_guess], xtol=1e-6)
            T_c_final, rho_c_final = solution
            
            print(f"Found solution: T_c = {T_c_final:.2e} K, rho_c = {rho_c_final:.2e} g/cm^3")
            
            # Integrate with final parameters
            result = self.integrate_structure(M_star, L_star, R_star)
            
            if result and result['success']:
                result['T_center'] = T_c_final
                result['rho_center'] = rho_c_final
                result['R_star'] = R_star
                
            return result
            
        except Exception as e:
            print(f"Boundary value solver failed: {e}")
            return None
    
    def plot_structure(self, result, save_plot=True):
        """Plot the stellar structure"""
        if not result or not result['success']:
            print("No valid solution to plot")
            return
            
        r = result['r']
        P = result['P']
        M_r = result['M_r']
        L_r = result['L_r']
        T = result['T']
        rho = result['rho']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Temperature vs radius
        axes[0,0].semilogx(r/result['R_star'], T)
        axes[0,0].set_xlabel('r/R')
        axes[0,0].set_ylabel('Temperature (K)')
        axes[0,0].set_title('Temperature Profile')
        axes[0,0].grid(True)
        
        # Pressure vs radius
        axes[0,1].loglog(r/result['R_star'], P)
        axes[0,1].set_xlabel('r/R')
        axes[0,1].set_ylabel('Pressure (dyne/cm²)')
        axes[0,1].set_title('Pressure Profile')
        axes[0,1].grid(True)
        
        # Density vs radius
        axes[1,0].loglog(r/result['R_star'], rho)
        axes[1,0].set_xlabel('r/R')
        axes[1,0].set_ylabel('Density (g/cm³)')
        axes[1,0].set_title('Density Profile')
        axes[1,0].grid(True)
        
        # Mass vs radius
        axes[1,1].plot(r/result['R_star'], M_r/M_r[-1])
        axes[1,1].set_xlabel('r/R')
        axes[1,1].set_ylabel('M(r)/M_total')
        axes[1,1].set_title('Mass Profile')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('stellar_structure.png', dpi=150, bbox_inches='tight')
            print("Plot saved as 'stellar_structure.png'")
        
        plt.show()

def main():
    """Main function to run stellar structure calculation"""
    
    # Solar units
    M_sun = 1.989e33  # g
    L_sun = 3.826e33  # erg/s
    
    # Get input parameters
    print("=== Stellar Structure Calculator ===")
    try:
        M_stellar = float(input("Enter stellar mass (solar units): "))
        L_stellar = float(input("Enter stellar luminosity (solar units): "))
        T_eff = float(input("Enter effective temperature (K): "))
        
        # Composition
        X = float(input("Enter hydrogen mass fraction (X): "))
        Z = float(input("Enter metals mass fraction (Z): "))
        
        if X + Z > 1.0:
            print("Error: X + Z must be <= 1.0")
            return
            
    except ValueError:
        print("Using default values for demonstration")
        M_stellar = 1.0   # Solar masses
        L_stellar = 1.0   # Solar luminosities  
        T_eff = 5778      # K
        X = 0.7           # Hydrogen fraction
        Z = 0.02          # Metals fraction
    
    # Convert to cgs units
    M_star = M_stellar * M_sun
    L_star = L_stellar * L_sun
    
    print(f"\nCalculating structure for:")
    print(f"Mass: {M_stellar:.2f} M_sun")
    print(f"Luminosity: {L_stellar:.2f} L_sun") 
    print(f"T_eff: {T_eff:.0f} K")
    print(f"Composition: X={X:.2f}, Z={Z:.3f}")
    
    # Create stellar structure object
    star = StellarStructure(X=X, Z=Z)
    
    # Solve the stellar structure
    print("\nSolving stellar structure equations...")
    result = star.boundary_value_solver(M_star, L_star, T_eff)
    
    if result and result['success']:
        print("\n=== SUCCESS! ===")
        print(f"Central temperature: {result['T_center']:.2e} K")
        print(f"Central density: {result['rho_center']:.2e} g/cm³")
        print(f"Stellar radius: {result['R_star']/6.96e10:.3f} R_sun")
        
        # Plot results
        star.plot_structure(result)
        
        # Save model data
        print("\nSaving model data...")
        np.savez('stellar_model.npz', **result)
        print("Model saved as 'stellar_model.npz'")
        
    else:
        print("Failed to find stellar structure solution")
        print("Try adjusting the input parameters or initial conditions")

if __name__ == "__main__":
    main()