import numpy as np


def INITCENTRAL(deltar, X, Z, mu, Rs, r_i, M_ri, L_ri, T_i, P_i ,cst):

       r = r_i + deltar
       M_rip1 = M_ri
       L_rip1 = L_ri
       T_ip1 = T_i
       P_ip1 = P_i
    
       return r, P_ip1, M_rip1, L_rip1, T_ip1


def   EOS(X, Z, XCNO, mu, P, T,izone,cst):

      Prad = cst.a*T**4/3.0e0
      Pgas = P - Prad

      rho=(mu*cst.m_H/cst.k_B)*(Pgas/T)
    
      if (T <= 0.0e0):
          print(f"Warning: Negative/zero temperature T={T} in zone {izone}")
          return (0.0, 0.0, 0.0, 0.0, 1)

    
      if (rho < 0.0e0):
          print(' I am sorry, but a negative density was detected.')
          print(' my equation-of-state routine is a bit baffled by this new')
          print(' physical system you have created.  The radiation pressure')
          print(' is probably too great, implying that the star is unstable.')
          print(' Please try something a little less radical next time.')
          print(' In case it helps, I detected the problem in zone ')
          print(' with the following conditions:')
          print('T       = {0:12.5E} K'.format(T))
          print('P_total = {0:12.5E} dynes/cm**2'.format(P))
          print('P_rad   = {0:12.5E} dynes/cm**2'.format(Prad))
          print('P_gas   = {0:12.5E} dynes/cm**2'.format(Pgas))
          print('rho     = {0:12.5E} g/cm**3'.format(rho))
          return (rho, 0.0 , 0.0 ,0.0,1)



      tog_bf = 2.82e0*(rho*(1.0e0 + X))**0.2e0
      k_bf = 4.34e25/tog_bf*Z*(1.0e0 + X)*rho/T**3.5e0
      k_ff = 3.68e22*cst.g_ff*(1.0e0 - Z)*(1.0e0 + X)*rho/T**3.5e0
      k_e = 0.2e0*(1.0e0 + X)
      kappa = k_bf + k_ff + k_e

      oneo3=0.333333333e0
      twoo3=0.666666667e0

      T6 = T*1.0e-06
      fx = 0.133e0*X*np.sqrt((3.0e0 + X)*rho)/T6**1.5e0
      fpp = 1.0e0 + fx*X
      psipp = 1.0e0 + 1.412e8*(1.0e0/X - 1.0e0)*np.exp(-49.98*T6**((-1.0)*oneo3))
      Cpp = 1.0e0 + 0.0123e0*T6**oneo3 + 0.0109e0*T6**twoo3 + 0.000938e0*T6
      epspp = 2.38e6*rho*X*X*fpp*psipp*Cpp*T6**(-twoo3)*np.exp(-33.80e0*T6**(-oneo3))
      CCNO = 1.0e0 + 0.0027e0*T6**oneo3 - 0.00778e0*T6**twoo3- 0.000149e0*T6
      epsCNO = 8.67e27*rho*X*XCNO*CCNO*T6**(-twoo3)*np.exp(-152.28e0*T6**(-oneo3))
      epslon = epspp + epsCNO

      return (rho, kappa, epslon, tog_bf,0)

def LUMINOBH(cst):
      Mpbh = 10e-11*1.989e33
      cs = 7.2*10e7
      k = 0.857
      T0 = 6.0e7
      ro=11.22
      Rbondi=2*cst.G*Mpbh/(cs)**2
      Ledd=(4*np.pi*cst.c*cst.G*Mpbh)/k
      Lbondi= 16*np.pi*0.1*ro*3*(cst.G*Mpbh)**2/(5*cs)
      Lbh=min(Ledd,Lbondi,1e27)
    
      return Lbh

def dPdr(r, M_r, rho,cst):
      return -cst.G*rho*M_r/r**2

def dMdr(r, rho, cst):
      return (4.0e0*np.pi*rho*r**2)

def dLdr(r, rho, epslon,cst):
      return (4.0e0*np.pi*rho*epslon*r**2)

def dTdr(r, M_r, L_r, T, rho, kappa, mu, irc,cst):
      if (irc == 0):
          return (-(3.0e0/(16.0e0*np.pi*cst.a*cst.c))*kappa*rho/T**3*L_r/r**2)
#  This is the adiabatic convective temperature gradient (Prialnik Eq. 6.29 or C&O Eq. 10.81).
      else:
          return (-1.0e0/cst.gamrat*cst.G*M_r/r**2*mu*cst.m_H/cst.k_B)

def RUNGE(f_im1, dfdr, r_im1, deltar, irc, X, Z, XCNO,mu, izone,cst):

      f_temp=np.zeros(4)
      f_i=np.zeros(4)

      dr12 = deltar/2.0e0
      dr16 = deltar/6.0e0
      r12  = r_im1 + dr12
      r_i  = r_im1 + deltar

      for i in range(0,4):
          f_temp[i] = f_im1[i] + dr12*dfdr[i]

      df1, ierr = FUNDEQ(r12, f_temp, irc, X, Z, XCNO, mu, izone,cst)
      if (ierr != 0):
          return f_i,ierr


      for i in range(0,4):
          f_temp[i] = f_im1[i] + dr12*df1[i]

      df2,ierr = FUNDEQ(r12, f_temp, irc, X, Z, XCNO, mu, izone,cst)
      if (ierr != 0):
          return f_i,ierr


      for i in range(0,4):
          f_temp[i] = f_im1[i] + deltar*df2[i]

      df3,ierr=FUNDEQ(r_i, f_temp, irc, X, Z, XCNO, mu, izone,cst)
      if (ierr != 0):
          return f_i,ierr

      for i in range(0,4):
          f_i[i] = f_im1[i] + dr16*(dfdr[i] + 2.0e0*df1[i] + 2.0e0*df2[i] + df3[i])

    # ✅ Safety check BEFORE returning
      for j in range(4):
          if not np.isfinite(f_i[j]) or f_i[j] <= 0.0:
              print(f"🛑 Runge-Kutta produced invalid value: f_i[{j}] = {f_i[j]}")
              return f_im1, 1
      
      return f_i,0


def FUNDEQ(r,f,irc,X,Z,XCNO,mu,izone,cst):

      dfdr=np.zeros(4)
      P   = f[0]
      M_r = f[1]
      L_r = f[2]
      T   = f[3]
      rho,kappa,epslon,tog_bf,ierr = EOS(X, Z, XCNO, mu, P, T, izone,cst)
      dfdr[0] = dPdr(r, M_r, rho,cst)
      dfdr[1] = dMdr(r, rho,cst)
      dfdr[2] = dLdr(r, rho, epslon,cst)
      dfdr[3] = dTdr(r, M_r, L_r, T, rho, kappa, mu, irc,cst)
      return (dfdr,ierr)


def StatStar(Msolar,Lsolar,Te,X,Z):


      nsh=999  

      r=np.zeros(nsh,float)
      P=np.zeros(nsh,float)
      M_r=np.zeros(nsh,float)
      L_r=np.zeros(nsh,float)
      T=np.zeros(nsh,float)
      rho=np.zeros(nsh,float)
      kappa=np.zeros(nsh,float)
      epslon=np.zeros(nsh,float)
      tog_bf=np.zeros(nsh,float)
      dlPdlT=np.zeros(nsh,float)



      deltar=0.0
      XCNO=0.0
      mu=0.0
      Ms=0.0
      Ls=0.0
      Rs=0.0
      T0=0.0
      P0=0.0
      Pcore=0.0
      Tcore=0.0
      rhocor=0.0
      epscor=0.0
      rhomax=0.0
      Rsolar=0.0

      Y=1.0-(X+Z)

      tog_bf0=0.01

      f_im1=np.zeros(4,float)
      dfdr=np.zeros(4,float)
      f_i=np.zeros(4,float)


      Nstart=20
      Nstop=999
      Igoof=-1
      ierr=0
      P0=0.0
      T0=0.0
      dlPlim=99.9
      debug=0

      Rsun=6.9599e10
      Msun=1.989e33
      Lsun=3.826e33

      class Constants:
           pass
      cst = Constants()

      cst.sigma=5.67051e-5
      cst.c=2.99792458e10
      cst.a=7.56591e-15
      cst.G=6.67259e-8
      cst.k_B=1.380658e-16
      cst.m_H=1.673534e-24
      cst.gamma= 5.0e0/3
      cst.g_ff= 1.0e0


      XCNO = Z/2.0e0

    
      Ms = Msolar*Msun
      Ls = Lsolar*Lsun
      Rs = np.sqrt(Ls/(4.e0*np.pi*cst.sigma))/Te**2
      Rsolar = Rs/Rsun

      deltar = Rs/100000.0e0


      mu = 1.0e0/(2.0*X + 0.75*Y + 0.5*Z)
      cst.gamrat = cst.gamma/(cst.gamma - 1.0e0)

      Mpbh = 1e-11*Msun
      cs = 7.2e7
      k = 0.857
      T0 = 5.0e7
      Rbondi=2*cst.G*Mpbh/(cs)**2
      ro=11.22
      P0 = 2.5e16
    

      Prad = cst.a * T0**4 / 3.0e0
      print(f"Initial T0 = {T0:.2e} K")
      print(f"Initial P0 = {P0:.2e} dyne/cm^2")
      print(f"Radiation pressure Prad = {Prad:.2e}")
      print(f"Gas pressure Pgas = {P0 - Prad:.2e}")

      
      initsh=0
      r[initsh]   = 5*Rbondi
      M_r[initsh] = Mpbh
      L_r[initsh] = LUMINOBH(cst)
      T[initsh]   = T0
      P[initsh]   = P0
      
      rho[initsh],kappa[initsh],epslon[initsh],tog_bf[initsh],ierr=EOS(X, Z, XCNO, mu, P[initsh], T[initsh], 0 ,cst)
      if ierr != 0:
          print("EOS failed at initial Bondi conditions. Aborting.")
          return

      tog_bf[0]=0.01
      dx = 0.05
      for i in range(0,20):
          ip1 = i + 1
          x = np.log(r[i])
          x1 = x + dx
          r[i+1] = np.exp(x1)
          dP = dPdr(r[i], M_r[i], rho[i], cst)
          dT = dTdr(r[i], M_r[i], L_r[i], T[i], rho[i], kappa[i], mu, 1, cst)
          dL = dLdr(r[i], rho[i], epslon[i], cst)
          dM = dMdr(r[i], rho[i], cst)
          # dy/dx = r * dy/dr 
          P[i+1]   = P[i]   + r[i] * dP * dx
          T[i+1]   = T[i]   + r[i] * dT * dx
          L_r[i+1] = L_r[i] + r[i] * dL * dx
          M_r[i+1] = M_r[i] + r[i] * dM * dx

          if P[i+1] <= 0.0 or T[i+1] <= 0.0:
              print(f"❌ Unphysical value at zone {i+1}: P = {P[i+1]:.3e}, T = {T[i+1]:.3e}")
              return 1, 1, i+1
        
          rho[i+1], kappa[i+1], epslon[i+1], tog_bf[i+1], ierr = EOS(X, Z, XCNO, mu, P[i+1], T[i+1], i+1, cst)
          if ierr != 0:
              print(f"EOS failed at zone {i+1}")
              return 1, 1, i+1
            
          print(f"[zone {i+1}] r = {r[i+1]:.4e}, T = {T[i+1]:.4e}, P = {P[i+1]:.4e}, rho = {rho[i+1]:.4e}")
 
    
      
      cst.kPad = 0.3e0
      irc = 0
      dlPdlT[initsh] = 4.25e0
 
      Nsrtp1 = ip1 + 1

      for i in range(Nsrtp1,Nstop):
          deltar = Rs/1000.0e0
          im1 = i - 1
          f_im1[0] = P[im1]
          f_im1[1] = M_r[im1]
          f_im1[2] = L_r[im1]
          f_im1[3] = T[im1]
          dfdr[0]  = dPdr(r[im1], M_r[im1], rho[im1],cst)
          dfdr[1]  = dMdr(r[im1], rho[im1],cst)
          dfdr[2]  = dLdr(r[im1], rho[im1], epslon[im1],cst)
          dfdr[3]  = dTdr(r[im1], M_r[im1], L_r[im1], T[im1], rho[im1],kappa[im1], mu, irc,cst)
          f_i,ierr=RUNGE(f_im1, dfdr, r[im1], deltar, irc, X, Z, XCNO, mu, i,cst)

          if (ierr != 0):
              print(' The problem occurred in the Runge-Kutta routine')
              print(' Values from the previous zone are:')
              print('r/Rs    = {0:12.5e}'.format(r[im1]/Rs))
              print('rho     = {0:12.5e} g/cm**3'.format(rho[im1]))
              print('M_r/Ms  = {0:12.5e}'.format(M_r[im1]/Ms))
              print('kappa   = {0:12.5e} cm**2/g'.format(kappa[im1]))
              print('T       = {0:12.5e} K'.format(T[im1]))
              print('epsilon = {0:12.5e} ergs/g/s'.format(epslon[im1]))
              print('P       = {0:12.5e} dynes/cm**2'.format(P[im1]))
              print('L_r/Ls  = {0:12.5e}'.format(L_r[im1]/Ls))
              break
#
#  Update stellar parameters for the next zone, including adding
#  dr to the old radius (note that dr <  0 since the integration is
#  inward).
#
          r[i]   = r[im1] + deltar
          P[i]   = f_i[0]
          M_r[i] = f_i[1]
          L_r[i] = f_i[2]
          T[i]   = f_i[3]
#
#  Calculate the density, opacity, and energy generation rate for
#  this zone.
#
          rho[im1],kappa[im1],epslon[im1],tog_bf[im1],ierr=EOS(X, Z, XCNO, mu, P[im1], T[i], i, cst)


          if (ierr != 0):
              print(' Values from the previous zone are:')
              print('r/Rs    = {0:12.5e}'.format(r[im1]/Rs))
              print('rho     = {0:12.5e} g/cm**3'.format(rho[im1]))
              print('M_r/Ms  = {0:12.5e}'.format(M_r[im1]/Ms))
              print('kappa   = {0:12.5e} cm**2/g'.format(kappa[im1]))
              print('T       = {0:12.5e} K'.format(T[im1]))
              print('epsilon = {0:12.5e} ergs/g/s'.format(epslon[im1]))
              print('P       = {0:12.5e} dynes/cm**2'.format(P[im1]))
              print('L_r/Ls  = {0:12.5e}'.format(L_r[im1]/Ls))
              istop = i
              break

          if (debug == 1): print (i,r[i],M_r[i],L_r[i],T[i],P[i],rho[i],kappa[i],epslon[i],tog_bf[i])



#
#  Determine whether convection will be operating in the next zone by
#  calculating dlnP/dlnT and comparing it to gamma/(gamma-1)
#  (see Prialnik Eq. 6.28 or C&O Eq. 10.87).  Set the convection flag appropriately.
#
          dlPdlT[i] = np.log(P[i]/P[im1])/np.log(T[i]/T[im1])
          if (dlPdlT[i] < cst.gamrat):
              irc = 1
          else:
              irc = 0


      grad_rad = (3 * kappa[0] * rho[0] * L_r[0]) / (16 * np.pi * cst.a * cst.c * T[0]**3 * r[0]**2)
      grad_conv = (1 / cst.gamrat) * cst.G * M_r[0] / r[0]**2 * mu * cst.m_H / cst.k_B

      print("dT/dr (radiative) =", -grad_rad)
      print("dT/dr (adiabatic) =", -grad_conv)
      print(f"Integrating from zone {Nsrtp1} to {Nstop}")

      if  (Igoof != 0):
          if (Igoof == -1):
              print('Sorry to be the bearer of bad news, but...')
              print('       Your model has some problems')
              print('The number of allowed shells has been exceeded')

          if (Igoof == 1):
              print('It looks like you are getting close,')
              print('however, there are still a few minor errors')
              print('The core density seems a bit off,')
              print(' density should increase smoothly toward the center.')
              print(' The density of the last zone calculated was rho = ',rho[istop],' gm/cm**3')
              print (rhocor,rhomax)
          if (rhocor > 1e10):
              print('It looks like you will need a degenerate')
              print(' neutron gas and general relativity')
              print(' to solve this core.  Who do you think I am, Einstein?')

          if (Igoof == 2):
              print('It looks like you are getting close,')
              print('however, there are still a few minor errors')
              print('The core epsilon seems a bit off,')
              print(' epsilon should vary smoothly near the center.')
              print(' The value calculated for the last zone was eps =',epslon[istop],' ergs/g/s')

          if (Igoof == 3):
              print('It looks like you are getting close,')
              print('however, there are still a few minor errors')
              print(' Your extrapolated central temperature is too low')
              print(' a little more fine tuning ought to do it.')
              print(' The value calculated for the last zone was T = ',T[istop],' K')

          if (Igoof == 4):
              print('Sorry to be the bearer of bad news, but...')
              print('       Your model has some problems')
              print('You created a star with a hole in the center!')

          if (Igoof == 5):
              print('Sorry to be the bearer of bad news, but...')
              print('       Your model has some problems')
              print('This star has a negative central luminosity!')

          if (Igoof == 6):
              print('Sorry to be the bearer of bad news, but...')
              print('       Your model has some problems')
              print('You hit the center before the mass and/or ')
              print('luminosity were depleted!')
      else:
          print('CONGRATULATIONS, I THINK YOU FOUND IT!')
          print('However, be sure to look at your model carefully.')

#
#  Print the central conditions.  If necessary, set limits for the
#  central radius, mass, and luminosity if necessary, to avoid format
#  field overflows.
#
      istop=i
      Rcrat = r[istop]/Rs
      if (Rcrat < -9.999e0): Rcrat = -9.999e0
      Mcrat = M_r[istop]/Ms
      if (Mcrat < -9.999e0): Mcrat = -9.999e0
      Lcrat = L_r[istop]/Ls
      if (Lcrat < -9.999e0): Lcrat = -9.999e0

      f=open('starmodl1661_py.dat','w')

      f.write('A Homogeneous Main-Sequence Model\n')
      f.write(' The surface conditions are:        The central conditions are:\n')
      f.write(' Mtot = {0:13.6E} Msun          Mc/Mtot     = {1:12.5E}\n'.format(Msolar,Mcrat))
      f.write(' Rtot = {0:13.6E} Rsun          Rc/Rtot     = {1:12.5E}\n'.format(Rsolar,Rcrat))
      f.write(' Ltot = {0:13.6E} Lsun          Lc/Ltot     = {1:12.5E}\n'.format(Lsolar,Lcrat))
      f.write(' Teff = {0:13.6E} K             Density     = {1:12.5E}\n'.format(Te,rhocor))
      f.write(' X    = {0:13.6E}               Temperature = {1:12.5E}\n'.format(X,Tcore))
      f.write(' Y    = {0:13.6E}               Pressure    = {1:12.5E} dynes/cm**2\n'.format(Y,Pcore))
      f.write(' Z    = {0:13.6E}               epsilon     = {1:12.5E} ergs/s/g\n'.format(Z,epscor))
      f.write('                                    dlnP/dlnT   = {0:12.5E}\n'.format(dlPdlT[istop]))

      f.write('Notes:\n')
      f.write(' (1) Mass is listed as Qm = 1.0 - M_r/Mtot, where Mtot = {0:13.6}\n'.format(Msun))
      f.write(' (2) Convective zones are indicated by c, radiative zones by r\n')
      f.write(' (3) dlnP/dlnT may be limited to +99.9 or -99.9# if so it is\n')
      f.write(' labeled by *\n')



#
#  Print data from the center of the star outward, labeling convective
#   or radiative zones by c or r, respectively.  If abs(dlnP/dlnT)
#  exceeds 99.9, set a print warning flag (*) and set the output limit
#  to +99.9 or -99.9 as appropriate to avoid format field overflows.
#
      f.write('   r        Qm       L_r       T        P        rho      kap      eps     dlPdlT\n')

      for ic in range(0,istop+1):
          i = istop - ic
          Qm = M_r[i]    # Total mass fraction down to radius


          if (dlPdlT[i] < cst.gamrat):
              rcf = 'c'
          else:
              rcf = 'r'
          if (np.abs(dlPdlT[i]) > dlPlim):
              dlPdlT[i] = np.copysign(dlPlim,dlPdlT[i])
              clim = '*'
          else:
              clim = ' '
          s='{0:7.5E} {1:7.5E} {2:7.5E} {3:7.5E} {4:7.5E} {5:7.5E} {6:7.5E} {7:6.5E}{8:1s}{9:1s} {10:5.1f}\n'.format(r[i], Qm, L_r[i], T[i], P[i], rho[i], kappa[i],epslon[i], clim, rcf, dlPdlT[i])
          f.write(s)

#     Output to screen
      print
      print('***** The integration has been completed *****')
      print('      The model has been stored in starmodl_py.dat')
      print
      return Igoof,ierr,istop

def main():

#
#  Enter desired stellar parameters
#


      getinp=1  # read in input
      if (getinp == 1):
           Msolar=float(input(' Enter the mass of the star (in solar units):'))
           Lsolar=float(input(' Enter the luminosity of the star (in solar units):'))
           Te=float(input(' Enter the effective temperature of the star (in K):'))
           Y=-1.0
           while (Y < 0.0):
               X=float(input(' Enter the mass fraction of hydrogen (X):'))
               Z=float(input(' Enter the mass fraction of metals (Z):'))
               Y = 1.e0 - X - Z
               if Y < 0:
                   print('You must have X + Z <= 1. Please reenter composition.')

      Igoof,ierr,istop=StatStar(Msolar,Lsolar,Te,X,Z)

main()

