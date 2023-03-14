import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from astropy import units as u
from astropy import constants as const
from astropy.modeling.blackbody import blackbody_nu
import extinction

# FIT FLUX SPECTRUM TO AN ACTIVE BINARY STAR SYSTEM WITH AN ACCRETION DISK
def main():
    # Collected data for binary star AM CVn: (Wavelength Angstroms, Flux Jy, Flux error Jy)
    AMCVn = ((4450 , 10.2e-3, 0.094e-3),(5510 , 8.43e-3, 0.078e-3),(6580 , 4.79e-3, 0.706e-3),(6730, 7.555e-3, 0.015e-3),(5320, 9.401e-3, 0.074e-3),(7970, 5.223e-3, 0.032e-3),(12350, 2.580e-3, 0.078e-3),(16620, 1.666e-3, 0.072e-3),(21590, 0.924e-3, 0.059e-3),(3543, 12.439e-3, 0.046e-3),(4770, 7.386e-3, 0.020e-3),(6231, 7.211e-3, 0.027e-3),(9134, 4.349e-3, 0.020e-3),(4810, 9.773e-3, 0.036e-3),(6170, 7.152e-3, 0.026e-3),(7520, 5.390e-3, 0.030e-3),(8660, 4.309e-3, 0.024e-3),(9620, 3.823e-3, 0.021e-3),(33526, 0.444e-3, 0.013e-3),(46028, 0.223e-3, 0.015e-3),(5822, 7.67e-03, 2.35E-05),(5036, 9.26e-03, 5.36E-05),(7620, 5.20e-03, 3.08E-05),(115608, 0.235e-3, 0),)
    
    data = []
    for wavelength, flux, flux_error in AMCVn:
        wavelength_grid = np.linspace(wavelength - 400, wavelength + 400, 200)
        p = np.exp(-((wavelength_grid - wavelength)/100))
        p[0] = p[-1] = 0

        # Add 10% errors in quadratic
        flux_error = np.sqrt((flux_error)**2 + (0.05 * flux)**2)
        data.append((wavelength_grid, p, wavelength, flux, flux_error))

    # Parameters and errors
    EBMV, EBMV_sig  = 0.03, 0.01 # Extinction: E(B-V)
    DISTANCE, DISTANCE_sig = 299.1, 4.4 # distance to star system in parsecs
    M1, M1_sig = 0.7, 0.08 # primary star mass
    ROUT   = 0.07 # disk outer radius
    NRAD   = 40 # number of annuli
    COSI   = 0.73 # projection factor
    MDOT   = 3e-8 # STARTING ACCERTION RATE


    ## COMPUTE BEST-FIT ACCRETION RATE
    min_wavlength, max_wavelength = 100000, 0
    for w, p, wc, f, fe in data:
        min_wavlength = min(min_wavlength, w.min())
        max_wavelength = max(max_wavelength, w.max())

    # Wavelength grid for spectrum integrals
    wavelength_grid = np.linspace(min_wavlength, max_wavelength, 500)

    # Fit accretion rates
    accretion_rates = []
    for _ in range(30):
        ebmv = np.random.normal(EBMV, EBMV_sig)
        if ebmv <= 0: continue
        av = 3.1*ebmv
        distance = np.random.normal(DISTANCE, DISTANCE_sig)
        m1 = np.random.normal(M1, M1_sig)

        # Radius of WD given its mass
        rin = eggleton_radius(m1)

        mdot = leastsq(res, [MDOT,], (m1, rin, ROUT, NRAD, distance, COSI, av, data, wavelength_grid), diag=[1.e-8,])[0][0]
        accretion_rates.append(np.log10(mdot))
        print('Computing best-fit accretion rate, Mdot = {:.5e}'.format(mdot))

    print('Accretion rate: log10(Mdot) = {:.2f} +/- {:.2f}'.format(np.array(accretion_rates).mean(), np.array(accretion_rates).std()))


    ## PLOT GRAPH
    plt.figure()
    plt.title('Reddened, black-body, steady-state accretion model for AM CVn')
    plt.xlabel('Wavelength [$\AA$]')
    plt.ylabel('Flux density [Jy]')

    # Plot simulated disk spectrum with applied extinction
    wave  = np.linspace(100, 125000, 500)
    dspec  = disc(wave, m1, mdot, rin, ROUT, NRAD, distance, COSI)
    dspec *= 10**(-extinction.fm07(wave, av))
    plt.plot(wave, dspec, 'b--')
  
    # Plot data points collected from AM CVn
    for w, p, wc, f, fe in data:
        plt.errorbar(wc, f, fe, color='r', ecolor='r', elinewidth=1, fmt='.k', markersize=7, capsize=2)

    # Write star/disk features on graph
    y1,y2 = plt.ylim()
    wl = 80000
    yp = 0.93*y2
    ys = 0.07*y2
    wr = wl+700
    plt.text(wl,yp,'$M_1 = {:.2f} \,\mathrm{{M}}_\odot$'.format(m1))
    yp -= ys
    emdot = int(math.floor(math.log10(mdot)))
    plt.text(wl,yp,r'$\dot{{M}} = {:.2f} \times 10^{{{:d}}}\,\mathrm{{M}}_\odot\,\mathrm{{yr}}^{{-1}}$'.format(mdot/10**emdot,emdot), color='b')

    yp -= ys
    plt.text(wl,yp,'$R_{{\mathrm{{in}}}} = {:.3f} \,\mathrm{{R}}_\odot$'.format(rin))

    yp -= ys
    plt.text(wl,yp,'$R_{{\mathrm{{out}}}} = {:.3f} \,\mathrm{{R}}_\odot$'.format(ROUT))

    yp -= ys
    plt.text(wl,yp,'$N_R = {:d}$'.format(NRAD))

    yp -= ys
    plt.text(wl,yp,'$d = {:d}\,\mathrm{{pc}}$'.format(int(round(distance))))

    yp -= ys
    plt.text(wl,yp,'$\cos(i) = {:.2f}$'.format(COSI))

    yp -= ys
    plt.text(wl,yp,'$E(B-V) = {:.2f} \,\mathrm{{mags}}$'.format(ebmv))

    plt.show()



## HELPER FUNCTIONS
# Radius of white dwarf (R sol) given its mass (M sol)
def eggleton_radius(mass):
    if isinstance(mass, np.ndarray):
        if (mass <= 0).any() or (mass >= 1.44).any():
            raise Exception("mr_wd_eggleton: mass array contains at least element outside range 0 to 1.44")
    elif mass <= 0 or mass >= 1.44:
        raise Exception("mr_wd_eggleton: mass = " + str(mass) + " out of range.")

    fac1 = np.power(mass/1.44,2./3.)
    fac2 = 0.00057/mass
    return 0.0114*np.sqrt(1./fac1-fac1)*np.power(1.+3.5*np.power(fac2,2./3.)+fac2,-2./3.)

# Computes flux density of disk (JY)
def disc(wave, m1, mdot, rin, rout, nrad, distance, cosi):
    # compute radii and temp of annuli spanning the disc
    R_ring = (rout-rin)/nrad
    ring_radii = np.linspace(rin+R_ring/2, rout-R_ring/2, nrad)
    ring_temps = tdisc(m1, rin, mdot, ring_radii)

    # Compute spectrum of each annuli
    for n, (R, T) in enumerate(zip(ring_radii, ring_temps)):
        if n == 0:
            spec = (const.R_sun*R)*(const.R_sun*R_ring)*blackbody_nu(wave*u.AA,T)
        else:
            spec += (const.R_sun*R)*(const.R_sun*R_ring)*blackbody_nu(wave*u.AA,T)

    # place at correct distance and account for projection
    spec *= 2*np.pi*cosi*u.sr/(distance*u.pc)**2

    return spec.to(u.Jy)/u.Jy

# Temperature of disk at radius R (R sol) around object mass M_star (M sol), radius R_star (R sol)
def tdisc(M_star, R_star, M_dot, radius):
    Tstar = (3*const.G*(const.M_sun*M_star)/(8*np.pi*const.sigma_sb) * (const.M_sun/u.yr*M_dot)/(const.R_sun*R_star)**3)**0.25
    return Tstar*(R_star/radius)**0.75 * (1-np.sqrt(R_star/radius))**0.25

def res(p, m1, rin, rout, nrad, distance, cosi, av, data, wgrid):
    mdot, = p

    sgrid = disc(wgrid, m1, mdot, rin, rout, nrad, distance, cosi)
    sgrid *= 10**(-extinction.fm07(wgrid, av))

    # interpolate and integrate over filter
    diffs = []
    for w,p,wc,f,fe in data:
        # interpolate onto filter wavelength grid
        spec = np.interp(w,wgrid,sgrid)

        # integrate (assumes uniform filter wavelength grids)
        pf = p/w**2
        model = (pf*spec).sum()/pf.sum()

        # store residual
        diffs.append((f-model)/fe)

    return diffs

# Computes flux density (Janskys) 
def star_spectrum(wavelength, temperature, radius, distance):
    spectrum = blackbody_nu(wavelength*u.AA, temperature)
    spectrum *= np.pi * (const.R_sun*radius/(const.pc*distance))**2*u.sr
    return spectrum.to(u.Jy)

main()