import numpy as np
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax_cosmo import Cosmology, background

fiducial_cosmo = {
    "Omega_c": 0.3,
    "Omega_b": 0.05,
    "h": 0.7,
    "sigma8": 0.8,
    "n_s": 0.96,
    "Omega_k": 0.,
    "w0": -1.0,
    "wa": 0.0,
}

@jax.jit
def get_cl_sigma8_omega_m(theta):

    cosmo = Cosmology(Omega_c=theta[0], Omega_b=0.05, h=0.7, sigma8=theta[1], n_s=0.96, Omega_k=0., w0=-1.0, wa=0.0)
    
    nz1 = jc.redshift.smail_nz(1., 2., 1.)
    nz2 = jc.redshift.smail_nz(1., 2., 0.5)

    nzs = [nz1, nz2]

    probes = [jc.probes.WeakLensing(nzs, sigma_e=0.26)]

    ell = np.logspace(1, 3)
    cls = jc.angular_cl.angular_cl(cosmo, ell, probes)

    return cls

@jax.jit
def get_cl(theta):

    cosmo = Cosmology(Omega_c=theta[0], Omega_b=theta[1], h=theta[2], sigma8=theta[3], n_s=theta[4], Omega_k=theta[5], w0=theta[6], wa=theta[7])

    nz1 = jc.redshift.smail_nz(1., 2., 1.)
    nz2 = jc.redshift.smail_nz(1., 2., 0.5)

    nzs = [nz1, nz2]

    probes = [jc.probes.WeakLensing(nzs, sigma_e=0.26)]

    ell = np.logspace(1, 3)
    cls = jc.angular_cl.angular_cl(cosmo, ell, probes)

    return cls

def get_prior(type="sigma_8_omega_m"):
    if type == "sigma_8_omega_m":
        num_params = 2
        minvals = np.array([0.1, 0.5])
        maxvals = np.array([0.8, 1.1])
    elif type == "all":
        num_params = 8
        minvals = np.array([0.1, 0.01, 0.5, 0.5, 0.8, -1.0, -1.5, -1])
        maxvals = np.array([0.8, 0.1, 1.0, 1.1, 1.2, 1.0, -0.5, 1])
    def prior(key, num_samples):
        return jax.random.uniform(key, shape=(num_samples, num_params), minval=minvals, maxval=maxvals)
    return prior