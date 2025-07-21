import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, minimize_scalar
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

class BlackHoleStarStructure:
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
        
        # Black hole parameters (will be set later)
        self.M_bh = None
        self.R_bondi = None
        
    def calculate_bondi_parameters(self, M_bh, cs=7.2e7, T_bondi=5.0e7, rho_bondi=11.22):
        """
        Calculate black hole and Bondi accretion parameters
        
        Parameters:
        M_bh: Black hole mass (g)
        cs: Sound speed (cm/s)
        T_bondi: Temperature at Bondi radius (K)
        rho_bondi: Density at Bondi radius (g/cm^3)
        """
        self.M_bh = M_bh
        self.cs = cs
        self.T_bondi = T_bondi
        self.rho_bondi = rho_bondi
        
        # Bondi radius
        self.R_bondi = 2 * self.cst.G * M_bh / cs**2
        
        # Eddington luminosity
        kappa_es = 0.34  # Thomson scattering opacity
        self.L_edd = 4 * np.pi * self.cst.c * self.cst.G * M_bh / kappa_es
        
        # Bondi accretion rate
        self.mdot_bondi = 4 * np.pi * rho_bondi * self.R_bondi**2 * cs
        
        # Black hole luminosity (taking minimum of Eddington and Bondi)
        eta = 0.1  # Radiative efficiency
        L_bondi = eta * self.mdot_bondi * self.cst.c**2
        self.L_bh = min(self.L_edd, L_bondi, 1e40)  # Cap at reasonable value
        
        # Pressure at Bondi radius from ideal gas law
        self.P_bondi = rho_bondi * self.cst.k_B * T_bondi / (self.mu * self.cst.m_H)
        
        print(f"Black hole parameters:")
        print(f"  M_bh = {M_bh/1.989e33:.2e} M_sun")
        print(f"  R_bondi = {self.R_bondi:.2e} cm = {self.R_bondi/6.96e10:.4f} R_sun")
        print(f"  L_bh = {self.L_bh:.2e} erg/s = {self.L_bh/3.826e33:.3f} L_sun")
        print(f"  L_edd = {self.L_edd:.2e} erg/s")
        print(f"  Bondi conditions:")
        print(f"    T = {T_bondi:.2e} K")
        print(f"    P = {self.P_bondi:.2e} dyne/cm^2")
        print(f"    rho = {rho_bondi:.2e} g/cm^3")
        
        return self.R_bondi, self.L_bh, self.P_bondi
        
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
            if T6 > 0:
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
            else:
                epspp = 0.0
                epsCNO = 0.0
                
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
        Stellar structure equations with black hole at center
        y = [P, M_r, L_r, T]
        """
        if r <= 0:
            return np.zeros(4)
            
        P, M_r, L_r, T = y
        
        # Get physical properties
        rho, kappa, epslon, tog_bf, ierr = self.equation_of_state(P, T)
        
        if ierr != 0 or rho <= 0:
            # Return small derivatives to prevent integration failure
            return np.array([1e-30, 1e-30, 1e-30, 1e-30])
        
        # Total mass includes black hole
        M_total = M_r + self.M_bh
        
        # Check for convection
        grad_rad = (3 * kappa * rho * L_r) / (16 * np.pi * self.cst.a * self.cst.c * T**3 * r**2)
        grad_ad = (1 / self.cst.gamrat) * self.cst.G * M_total / r**2 * self.mu * self.cst.m_H / self.cst.k_B
        
        # Choose appropriate temperature gradient
        if grad_rad > grad_ad and L_r > 0:  # Convective
            dT_dr = -grad_ad
            conv_flag = True
        else:  # Radiative
            dT_dr = -grad_rad if L_r > 0 else -grad_ad
            conv_flag = False
            
        # Structure equations (modified for black hole)
        dP_dr = -self.cst.G * rho * M_total / r**2
        dM_dr = 4.0 * np.pi * rho * r**2  # Only stellar mass, not including BH
        dL_dr = 4.0 * np.pi * rho * epslon * r**2
        
        return np.array([dP_dr, dM_dr, dL_dr, dT_dr])
    
    def get_bondi_initial_conditions(self):
        """
        Set up initial conditions at the Bondi radius
        """
        if self.R_bondi is None:
            raise ValueError("Must call calculate_bondi_parameters first!")
            
        # Initial stellar mass (just the matter outside Bondi radius)
        M_stellar_bondi = 0.0  # Start with no stellar mass at Bondi radius
        
        # Initial conditions at Bondi radius: [P, M_r, L_r, T]
        y0 = np.array([
            self.P_bondi,      # Pressure
            M_stellar_bondi,   # Stellar mass (excluding BH)
            self.L_bh,         # Luminosity from BH accretion
            self.T_bondi       # Temperature
        ])
        
        return self.R_bondi, y0
    
    def integrate_bh_star_structure(self, R_star, verbose=True):
        """
        Integrate stellar structure from Bondi radius outward
        """
        if self.R_bondi is None:
            raise ValueError("Must set up black hole parameters first!")
            
        if verbose:
            print(f"Integrating from Bondi radius to stellar surface:")
            print(f"R_bondi = {self.R_bondi:.2e} cm")
            print(f"R_star = {R_star:.2e} cm")
        
        # Get initial conditions at Bondi radius
        r_start, y0 = self.get_bondi_initial_conditions()
        
        if R_star <= r_start:
            print("Error: Stellar radius must be larger than Bondi radius!")
            return None
            
        # Integration span (outward from Bondi radius)
        r_span = (r_start, R_star)
        
        # Use logarithmic spacing for better resolution near Bondi radius
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
                max_step=(R_star - r_start)/100,  # Reasonable max step
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
            M_r = sol.y[1]  # Stellar mass (excluding BH)
            L_r = sol.y[2]
            T = sol.y[3]
            
            # Calculate derived quantities
            n_points = len(r)
            rho = np.zeros(n_points)
            kappa = np.zeros(n_points)
            epslon = np.zeros(n_points)
            M_total = np.zeros(n_points)  # Total mass including BH
            
            for i in range(n_points):
                rho[i], kappa[i], epslon[i], _, _ = self.equation_of_state(P[i], T[i])
                M_total[i] = M_r[i] + self.M_bh
            
            return {
                'r': r,
                'P': P,
                'M_r': M_r,           # Stellar mass only
                'M_total': M_total,   # Total mass (stellar + BH)
                'L_r': L_r,
                'T': T,
                'rho': rho,
                'kappa': kappa,
                'epslon': epslon,
                'R_bondi': self.R_bondi,
                'M_bh': self.M_bh,
                'L_bh': self.L_bh,
                'success': True
            }
            
        except Exception as e:
            print(f"Integration error: {e}")
            return None
    
    def find_stellar_radius(self, M_star_target, L_star_target, T_eff_target, 
                          R_guess=None, verbose=True):
        """
        Find stellar radius that gives desired surface conditions
        """
        if R_guess is None:
            # Initial guess based on Stefan-Boltzmann law
            R_guess = np.sqrt(L_star_target / (4 * np.pi * self.cst.sigma)) / T_eff_target**2
            
        if verbose:
            print(f"Finding stellar radius for target conditions:")
            print(f"M_star = {M_star_target/1.989e33:.2f} M_sun")
            print(f"L_star = {L_star_target/3.826e33:.2f} L_sun")
            print(f"T_eff = {T_eff_target:.0f} K")
            print(f"Initial radius guess: {R_guess/6.96e10:.3f} R_sun")
        
        def objective(log_R):
            """Objective function to minimize"""
            R_trial = np.exp(log_R)
            
            if R_trial <= self.R_bondi * 1.1:  # Must be larger than Bondi radius
                return 1e10
                
            result = self.integrate_bh_star_structure(R_trial, verbose=False)
            
            if not result or not result['success']:
                return 1e10
                
            # Check surface conditions
            P_surface = result['P'][-1]
            M_surface = result['M_r'][-1]  # Stellar mass at surface
            L_surface = result['L_r'][-1]
            T_surface = result['T'][-1]
            
            # Calculate effective temperature from luminosity and radius
            T_eff_calc = (L_surface / (4 * np.pi * self.cst.sigma * R_trial**2))**0.25
            
            # Objective: minimize differences from targets
            # Weight different terms appropriately
            mass_error = abs(M_surface - M_star_target) / M_star_target
            lum_error = abs(L_surface - L_star_target) / L_star_target  
            temp_error = abs(T_eff_calc - T_eff_target) / T_eff_target
            pressure_penalty = P_surface / 1e10  # Surface pressure should be small
            
            total_error = mass_error + lum_error + temp_error + pressure_penalty
            
            if verbose:
                print(f"R = {R_trial/6.96e10:.3f} R_sun, errors: M={mass_error:.3f}, "
                      f"L={lum_error:.3f}, T={temp_error:.3f}, total={total_error:.3f}")
            
            return total_error
        
        try:
            # Use bounded optimization
            result = minimize_scalar(
                objective, 
                bounds=(np.log(self.R_bondi * 1.2), np.log(R_guess * 10)),
                method='bounded'
            )
            
            if result.success:
                R_optimal = np.exp(result.x)
                if verbose:
                    print(f"Found optimal radius: {R_optimal/6.96e10:.3f} R_sun")
                
                # Return full stellar structure
                return self.integrate_bh_star_structure(R_optimal, verbose)
            else:
                print("Optimization failed to converge")
                return None
                
        except Exception as e:
            print(f"Radius optimization error: {e}")
            return None
    
    def plot_bh_star_structure(self, result, save_plot=True):
        """Plot the stellar structure with black hole"""
        if not result or not result['success']:
            print("No valid solution to plot")
            return
            
        r = result['r']
        P = result['P']
        M_r = result['M_r']
        M_total = result['M_total']
        L_r = result['L_r']
        T = result['T']
        rho = result['rho']
        R_bondi = result['R_bondi']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Normalize radius
        r_norm = r / r[-1]  # Normalize by stellar radius
        bondi_norm = R_bondi / r[-1]
        
        # Temperature vs radius
        axes[0,0].semilogx(r_norm, T)
        axes[0,0].axvline(bondi_norm, color='red', linestyle='--', alpha=0.7, label='Bondi radius')
        axes[0,0].set_xlabel('r/R_star')
        axes[0,0].set_ylabel('Temperature (K)')
        axes[0,0].set_title('Temperature Profile')
        axes[0,0].grid(True)
        axes[0,0].legend()
        
        # Pressure vs radius
        axes[0,1].loglog(r_norm, P)
        axes[0,1].axvline(bondi_norm, color='red', linestyle='--', alpha=0.7, label='Bondi radius')
        axes[0,1].set_xlabel('r/R_star')
        axes[0,1].set_ylabel('Pressure (dyne/cm²)')
        axes[0,1].set_title('Pressure Profile')
        axes[0,1].grid(True)
        axes[0,1].legend()
        
        # Density vs radius
        axes[0,2].loglog(r_norm, rho)
        axes[0,2].axvline(bondi_norm, color='red', linestyle='--', alpha=0.7, label='Bondi radius')
        axes[0,2].set_xlabel('r/R_star')
        axes[0,2].set_ylabel('Density (g/cm³)')
        axes[0,2].set_title('Density Profile')
        axes[0,2].grid(True)
        axes[0,2].legend()
        
        # Mass vs radius (both stellar and total)
        axes[1,0].plot(r_norm, M_r/1.989e33, label='Stellar mass', linewidth=2)
        axes[1,0].plot(r_norm, M_total/1.989e33, label='Total mass (stellar + BH)', linewidth=2)
        axes[1,0].axvline(bondi_norm, color='red', linestyle='--', alpha=0.7, label='Bondi radius')
        axes[1,0].set_xlabel('r/R_star')
        axes[1,0].set_ylabel('Mass (M_sun)')
        axes[1,0].set_title('Mass Profile')
        axes[1,0].grid(True)
        axes[1,0].legend()
        
        # Luminosity vs radius
        axes[1,1].semilogx(r_norm, L_r/3.826e33)
        axes[1,1].axvline(bondi_norm, color='red', linestyle='--', alpha=0.7, label='Bondi radius')
        axes[1,1].set_xlabel('r/R_star')
        axes[1,1].set_ylabel('Luminosity (L_sun)')
        axes[1,1].set_title('Luminosity Profile')
        axes[1,1].grid(True)
        axes[1,1].legend()
        
        # Mass-radius relation
        axes[1,2].plot(r/6.96e10, M_total/1.989e33, linewidth=2, label='Total mass')
        axes[1,2].plot(r/6.96e10, M_r/1.989e33, linewidth=2, label='Stellar mass')
        axes[1,2].axvline(R_bondi/6.96e10, color='red', linestyle='--', alpha=0.7, label='Bondi radius')
        axes[1,2].set_xlabel('Radius (R_sun)')
        axes[1,2].set_ylabel('Mass (M_sun)')
        axes[1,2].set_title('Mass-Radius Relation')
        axes[1,2].grid(True)
        axes[1,2].legend()
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('blackhole_star_structure.png', dpi=150, bbox_inches='tight')
            print("Plot saved as 'blackhole_star_structure.png'")
        
        plt.show()

def main():
    """Main function for black hole + star structure calculation"""
    
    # Physical constants
    M_sun = 1.989e33  # g
    L_sun = 3.826e33  # erg/s
    
    print("=== Black Hole + Stellar Structure Calculator ===")
    
    try:
        # Black hole parameters
        print("\nBlack hole parameters:")
        M_bh_solar = float(input("Enter black hole mass (solar masses): "))
        
        print("\nBondi accretion parameters:")
        cs = float(input("Sound speed at Bondi radius (cm/s, default=7.2e7): ") or 7.2e7)
        T_bondi = float(input("Temperature at Bondi radius (K, default=5e7): ") or 5e7)
        rho_bondi = float(input("Density at Bondi radius (g/cm³, default=11.22): ") or 11.22)
        
        print("\nStellar parameters:")
        M_stellar = float(input("Target stellar mass (solar units): "))
        L_stellar = float(input("Target stellar luminosity (solar units): "))
        T_eff = float(input("Target effective temperature (K): "))
        
        # Composition
        print("\nStellar composition:")
        X = float(input("Hydrogen mass fraction (X, default=0.7): ") or 0.7)
        Z = float(input("Metals mass fraction (Z, default=0.02): ") or 0.02)
        
        if X + Z > 1.0:
            print("Error: X + Z must be <= 1.0")
            return
            
    except (ValueError, KeyboardInterrupt):
        print("\nUsing default values for demonstration")
        M_bh_solar = 1e-5     # Very small BH for testing
        cs = 7.2e7
        T_bondi = 5e7
        rho_bondi = 11.22
        M_stellar = 1.0       # Solar masses
        L_stellar = 1.0       # Solar luminosities  
        T_eff = 5778          # K
        X = 0.7               # Hydrogen fraction
        Z = 0.02              # Metals fraction
    
    # Convert to cgs units
    M_bh = M_bh_solar * M_sun
    M_star = M_stellar * M_sun
    L_star = L_stellar * L_sun
    
    print(f"\nSystem parameters:")
    print(f"Black hole mass: {M_bh_solar:.2e} M_sun")
    print(f"Target stellar mass: {M_stellar:.2f} M_sun")
    print(f"Target luminosity: {L_stellar:.2f} L_sun") 
    print(f"Target T_eff: {T_eff:.0f} K")
    print(f"Composition: X={X:.2f}, Z={Z:.3f}")
    
    # Create black hole + star structure object
    bh_star = BlackHoleStarStructure(X=X, Z=Z)
    
    # Set up black hole and Bondi parameters
    print("\nCalculating Bondi parameters...")
    R_bondi, L_bh, P_bondi = bh_star.calculate_bondi_parameters(
        M_bh, cs, T_bondi, rho_bondi
    )
    
    # Find stellar structure
    print(f"\nSolving for stellar structure...")
    print("This may take a moment as we search for the optimal stellar radius...")
    
    result = bh_star.find_stellar_radius(M_star, L_star, T_eff, verbose=True)
    
    if result and result['success']:
        print("\n=== SUCCESS! ===")
        
        # Extract final properties
        R_star = result['r'][-1]
        M_stellar_final = result['M_r'][-1]
        M_total_final = result['M_total'][-1]
        L_final = result['L_r'][-1]
        T_final = result['T'][-1]
        P_surface = result['P'][-1]
        
        # Calculate effective temperature
        T_eff_calc = (L_final / (4 * np.pi * bh_star.cst.sigma * R_star**2))**0.25
        
        print(f"\nFinal stellar structure:")
        print(f"Stellar radius: {R_star/6.96e10:.3f} R_sun")
        print(f"Stellar mass: {M_stellar_final/M_sun:.3f} M_sun")
        print(f"Total mass (star+BH): {M_total_final/M_sun:.3f} M_sun")
        print(f"Luminosity: {L_final/L_sun:.3f} L_sun")
        print(f"Effective temperature: {T_eff_calc:.0f} K")
        print(f"Surface pressure: {P_surface:.2e} dyne/cm²")
        
        print(f"\nBondi radius: {R_bondi/6.96e10:.4f} R_sun")
        print(f"R_star/R_bondi ratio: {R_star/R_bondi:.1f}")
        
        # Plot results
        bh_star.plot_bh_star_structure(result)
        
        # Save model data
        print("\nSaving model data...")
        result['input_parameters'] = {
            'M_bh': M_bh,
            'M_stellar_target': M_star,
            'L_stellar_target': L_star,
            'T_eff_target': T_eff,
            'X': X, 'Z': Z,
            'cs': cs,
            'T_bondi': T_bondi,
            'rho_bondi': rho_bondi
        }
        np.savez('blackhole_stellar_model.npz', **result)
        print("Model saved as 'blackhole_stellar_model.npz'")
        
    else:
        print("Failed to find stellar structure solution")
        print("Try adjusting the black hole mass or stellar parameters")

if __name__ == "__main__":
    main()