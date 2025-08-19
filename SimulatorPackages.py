import pandas as pd
from random import sample
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from astropy.cosmology import FlatLambdaCDM
from tqdm import tqdm
import os
import copy

import gwfast.gwfastGlobals as glob
from gwfast.waveforms import IMRPhenomD_NRTidalv2, TaylorF2_RestrictedPN, IMRPhenomD
from gwfast.signal import GWSignal
from gwfast.network import DetNet
from fisherTools import CovMatr, compute_localization_region, check_covariance, fixParams

import jax

print("Available devices:", jax.devices())

# Ensure JAX is using GPU
assert any(device.platform == "gpu" for device in jax.devices()), "GPU is not being used!"


class GWSampler:
    def __init__(self, host_g = pd.DataFrame({})):
        """Initialization"""
        self.host_g = host_g

        self.BNS  = pd.DataFrame({})
        self.BBH  = pd.DataFrame({})
        self.NSBH = pd.DataFrame({})

    # Mass distributions
    def stepf(self, m1,Mmin,Mmax, val1 = 0, val2 = 0):
      return np.heaviside(m1-Mmin, val1) * np.heaviside(Mmax - m1, val2)

    def ff(self,m,delta):
      return (1 + np.exp(delta/m + delta/(m - delta)))**(-1)

    def Smoothf(self,m1,Mmin,delta):
      filtr1 = m1 > Mmin
      filtr2 = m1 < Mmin + delta
      filtr3 = np.logical_and(filtr1,filtr2)

      res = np.concatenate([np.zeros(np.sum(~filtr1)),
                            self.stepf(m1[filtr3],Mmin,Mmin+delta) * self.ff(m1[filtr3]-Mmin,delta),
                            np.ones(np.sum(~filtr2))])
      return res

    def trunc_plaw(self,m1,Mmin,Mmax,alpha,val1 = 0, val2 = 0):
      return self.stepf(m1,Mmin,Mmax,val1,val2) * m1**(-alpha)

    def broken_plaw(self,m1,Mmin,Mmax,alpha1,alpha2,b,delta_m):
      m_break = Mmin + b * (Mmax - Mmin)
      return self.trunc_plaw(m1,Mmin,m_break,alpha1) + self.trunc_plaw(m1,m_break,Mmax,alpha2) * self.trunc_plaw(m_break,Mmin,m_break,alpha1,val2=1) / self.trunc_plaw(m_break,m_break,Mmax,alpha2,val1=1)
      # return Smoothf(m1,Mmin,delta_m) * (stepf(m1,Mmin,m_break,val2=1) * m1**(-alpha1) + stepf(m1,m_break,Mmax) * m1**(-alpha2))

    def gaussian_plaw(self,m1,Mmin,Mmax,alpha,mu,sigma,lam,delta_m):
      ys = self.Smoothf(m1,Mmin,delta_m) * self.stepf(m1,Mmin,Mmax) * ((1 - lam) * self.trunc_plaw(m1,Mmin,Mmax,alpha) + lam * norm(loc = mu, scale = sigma).pdf(m1))
      return np.where(ys > 1e-16, ys, 0)

    def multipeak_plaw(self,m1,Mmin,Mmax,alpha,mu,sigma,lam,delta_m,mu_l,sigma_l,lam_l):
      return self.Smoothf(m1,Mmin,delta_m) * self.stepf(m1,Mmin,Mmax) * ((1 - lam) * self.trunc_plaw(m1,Mmin,Mmax,alpha) + lam * (1 - lam_l) * norm(loc = mu, scale = sigma).pdf(m1) + lam * lam_l * norm(loc = mu_l, scale = sigma_l).pdf(m1))

    # def pm2(m2,Mmin,m1,beta):
    #   return trunc_plaw(m2,Mmin,m1,-beta)
    # Taken from pycbc documentation https://pycbc.org/pycbc/latest/html/_modules/pycbc/conversions.html
    def eta(self, mass1, mass2):
        """Returns the symmetric mass ratio from mass1 and mass2."""
        return mass1*mass2 / (mass1 + mass2)**2.

    def mchirp(self, mass1, mass2):
        """Returns the chirp mass from mass1 and mass2."""
        return self.eta(mass1, mass2)**(3./5) * (mass1 + mass2)


    def sample_m(self, n = 1000, profile = 'gaussian', args = None):

        if args == None:
          if profile == 'trunc':
            profilef  = self.trunc_plaw
            args      = (1,100,1.5)

          elif profile == 'broken':
            profilef  = self.broken_plaw
            args      = (1,100,1,2,0.2,1)

          elif profile == 'gaussian':
            profilef  = self.gaussian_plaw
            args      = (2.5,100,3.4,35,3.9,0.04,4.8)

          elif profile == 'multipeak':
            profilef  = self.multipeak_plaw
            args      = (1,100,2,8,3,0.3,1,30,4,0.3)

        else:
          if profile == 'trunc':
            profilef  = self.trunc_plaw

          elif profile == 'broken':
            profilef  = self.broken_plaw

          elif profile == 'gaussian':
            profilef  = self.gaussian_plaw

          elif profile == 'multipeak':
            profilef  = self.multipeak_plaw


        Mmin, Mmax = args[:2]

        # Define the custom PDF
        def pdf(x):
            return profilef(x,*args)
            # return trunc_plaw(x,Mmin,Mmax,1.5)
            # return broken_plaw(x,Mmin,Mmax,alpha1 = 1,alpha2 = 2,b = 0.2, delta_m = 1)
            # return gaussian_plaw(x,Mmin,Mmax,alpha = 1,mu = 8,sigma = 3,lam = 0.9, delta_m = 0.5)
            # return multipeak_plaw(x,Mmin,Mmax,alpha = 2,mu = 8,sigma = 3,lam = 0.3,delta_m = 1,mu_l = 30,sigma_l = 4,lam_l = 0.3)


        # Create a grid for x
        m1 = np.geomspace(Mmin, Mmax, 1000000)

        # Compute the normalized CDF over the grid
        cdf_values = cumulative_trapezoid(pdf(m1), m1, initial=0)  # Integrate the PDF
        cdf_values /= cdf_values[-1]  # Normalize to make it a valid CDF
        fin_zero_ind = np.sum(cdf_values == 0) - 1
        cdf_values = cdf_values[fin_zero_ind:]
        m1 = m1[fin_zero_ind:]

        # Interpolate the inverse CDF using CubicSpline
        inverse_cdf = CubicSpline(cdf_values, m1, bc_type='clamped')

        # Generate uniform samples
        u_samples = np.random.uniform(0, 1, n)

        # Map uniform samples through the inverse CDF
        samples = inverse_cdf(u_samples)

        return samples
    

    def reset_samples(self):
      self.BNS  = pd.DataFrame({})
      self.BBH  = pd.DataFrame({})
      self.NSBH = pd.DataFrame({})

    def sample_BBH(self,n = 1000):
      
      if not self.host_g.empty:
        print('Change n to comply with host_g.')
        n = len(self.host_g)
        self.BBH = self.host_g.copy()
        # self.BBH['dL'] = self.host_g['dL'].copy()

      samples_m1 = self.sample_m(n = n)
      samples_m2 = np.random.uniform(low=2.5, high = samples_m1)
      self.BBH['m1'] = samples_m1
      self.BBH['m2'] = samples_m2

      # Add eta and Mchirp
      self.BBH['eta'] = self.eta(self.BBH['m1'],self.BBH['m2'])
      self.BBH['Mc']  = self.mchirp(self.BBH['m1'],self.BBH['m2'])

      # Add other parameters
      self.BBH['iota']     = np.arccos(np.random.uniform(-1., 1., n))
      self.BBH['psi']      = np.random.uniform(0., 2.*np.pi, n)
      self.BBH['tcoal']    = np.random.uniform(0., 1., n)
      self.BBH['Phicoal']  = np.random.uniform(0., 2.*np.pi, n)
      # self.BBH['chi1z']    = np.random.uniform(-.05, .05, n)
      # self.BBH['chi2z']    = np.random.uniform(-.05, .05, n)
      self.BBH['chi1z']    = np.random.uniform(-.9, .9, n)
      self.BBH['chi2z']    = np.random.uniform(-.9, .9, n)
      
      if not self.host_g.empty:
        self.host_g[self.BBH.keys()] = self.BBH
      return None

    def sample_BNS(self,n = 1000):
      print('Not yet supported.')
      # if not self.host_g.empty:
      #   print('Change n to comply with host_g.')
      #   n = len(self.host_g)
      
      # samples_m1 = np.random.uniform(low=2.5, high = samples_m1)
      # self.sample_m(n = n)
      # samples_m2 = np.random.uniform(low=2.5, high = samples_m1)
      # self.BBH['m1'] = samples_m1
      # self.BBH['m2'] = samples_m2
      return None


    def sample_NSBH(self,n = 1000):
      print('Not yet supported.')
      # if not self.host_g.empty:
      #   print('Change n to comply with host_g.')
      #   n = len(self.host_g)
      
      # samples_m1 = self.sample_m(n = n)
      # samples_m2 = np.random.uniform(low=1, high = 2.5)
      # self.NSBH['m1'] = samples_m1
      # self.NSBH['m2'] = samples_m2
      return None

################################################################################
################################################################################
################################################################################

class RedshiftSampler():
    def __init__(self, zmin = 0, zmax = 5, H0=70, Om0=0.3, Ob0=0.044):
      """Initialization"""
      self.cosmo  = FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0)
      self.zmin, self.zmax = zmin, zmax
      self.R0     = 30. #(BBH) in Gpc^-3 yr^-1
      self.alpha  = 1.
      self.beta   = 3.4
      self.z_p    = 2.4
      self.host_z = pd.DataFrame({})

    # comoving merger rate
    def R(self, z):
      # Gpc^-3 yr^-1
      return self.R0 * (1+z)**self.alpha / (1 + ((1 + z)/(1 + self.z_p))**(self.alpha + self.beta))

    # Number of gw events
    def dN_gw(self, z, dz, dt):
      # dt in yr
      dVc_dz = self.cosmo.differential_comoving_volume(z).value * 1e-9
      return self.R(z) / (1+z) * dVc_dz * 4*np.pi * dz * dt

    def sample_z(self, n = 1000, dt = 1, give_N_tot = False):
      zs = np.linspace(self.zmin,self.zmax,100000)
      dz = zs[1] - zs[0]

      dN_gws = self.dN_gw(z = zs, dz = dz, dt = dt)

      # Compute the normalized CDF over the grid
      cdf_values = cumulative_trapezoid(dN_gws, zs, initial=0)  # Integrate the PDF

      if give_N_tot:
        n = int(np.sum(dN_gws))

      cdf_values /= cdf_values[-1]  # Normalize to make it a valid CDF

      # Interpolate the inverse CDF using CubicSpline
      inverse_cdf = CubicSpline(cdf_values, zs, bc_type='clamped')

      # Generate uniform samples
      u_samples = np.random.uniform(0, 1, n)

      # Map uniform samples through the inverse CDF
      samples = inverse_cdf(u_samples)

      # Save samples
      self.host_z['z']  = samples
      self.host_z['dL'] = self.cosmo.luminosity_distance(samples).value * 1e-3 # Gpc
      # Add random position on the sky
      self.host_z['ra']  = np.random.uniform(0, 2*np.pi, len(samples)) # rad
      self.host_z['dec'] = np.arcsin(np.random.uniform(-1, 1, len(samples))) # rad

      return None


class EventSimulator:
    def __init__(self, host_g, filt_snr = None, filt_err = None, batch = 50):
        self.new_cols = ['snrs','PASS','theta','phi','inversion_err','skyArea','sigma_Mc','sigma_eta','sigma_dL','sigma_theta','sigma_phi']
        self.new_cols_snr_only = ['snrs']
        self.host_g   = host_g
        self.filt_snr = filt_snr
        self.filt_err = filt_err
        self.batch    = batch
        ########################################################################
        alldetectors = copy.deepcopy(glob.detectors)
        print('All available detectors are: '+str(list(alldetectors.keys())))
    
        # select 1ET+2CE
        MYdetectors = {det:alldetectors[det] for det in ['ETMR', 'CE1Id', 'CE2NM']}
        
        # # select only ET
        # MYdetectors = {det:alldetectors[det] for det in ['ETMR']}
        
        print('Using detectors '+str(list(MYdetectors.keys())))
    
        # We use the O2 psds
        MYdetectors['ETMR']['psd_path']  = os.path.join(glob.detPath, 'ET-0000A-18.txt')
        MYdetectors['CE1Id']['psd_path'] = os.path.join(glob.detPath, 'ce_strain/cosmic_explorer_20km.txt')
        MYdetectors['CE2NM']['psd_path'] = os.path.join(glob.detPath, 'ce_strain/cosmic_explorer_20km.txt')
    
        mySignalsET = {}

        for d in MYdetectors.keys():
            mySignalsET[d] = GWSignal(
                # TaylorF2_RestrictedPN(use_3p5PN_SpinHO=True, is_tidal=True),
                IMRPhenomD(),
                # (True, True),
                        psd_path= MYdetectors[d]['psd_path'],
                        detector_shape = MYdetectors[d]['shape'],
                        det_lat= MYdetectors[d]['lat'],
                        det_long=MYdetectors[d]['long'],
                        det_xax=MYdetectors[d]['xax'],
                        verbose=True,
                        useEarthMotion = False,
                        fmin=2.,
                        IntTablePath=None)

        self.myNet = DetNet(mySignalsET)
        ########################################################################
    
    def translate(self, df, filt = np.array([])):

        if filt.size == 0:
          filt = np.full(len(df), True)
        # zs       = df['z'].values[filt]
        zs       = df['true_redshift_gal'].values[filt]
        dLs      = df['dL'].values[filt]
        Mcs      = df['Mc'].values[filt]
        etas     = df['eta'].values[filt]
        RAs      = df['ra'].values[filt]
        DECs     = df['dec'].values[filt]
        iotas    = df['iota'].values[filt]
        psis     = df['psi'].values[filt]
        tcoals   = df['tcoal'].values[filt]
        Phicoals = df['Phicoal'].values[filt]
        chi1zs   = df['chi1z'].values[filt]
        chi2zs   = df['chi2z'].values[filt]

        events = {'Mc'       : Mcs*(1.+zs),
                  'eta'      : etas,
                  'dL'       : dLs,
                  'ra'       : RAs,
                  'dec'      : DECs,
                  'iota'     : iotas,
                  'psi'      : psis,
                  'tcoal'    : tcoals,
                  'Phicoal'  : Phicoals,
                  'chi1z'    : chi1zs,
                  'chi2z'    : chi2zs,
                  'Lambda1'  : np.zeros(np.sum(filt)),
                  'Lambda2'  : np.zeros(np.sum(filt))
                  }
        return events


    def simulate_batch(self, df, snr_only = False):

        nevents = len(df)

        events = self.translate(df.copy())

        ####### SNR
        snrs = self.myNet.SNR(events)
        df['snrs'] = snrs

        if self.filt_snr != None:
          filtered_snr = snrs > self.filt_snr
          print(np.sum(~filtered_snr), 'out of', len(snrs),'dropped by SNR:',np.sum(filtered_snr), 'passed.')
          
          snrs = snrs[filtered_snr]
          events = self.translate(df.copy(), filtered_snr)
        else:
          filtered_snr = np.full(nevents, True)

        ####### INVERSION ERROR
        if not snr_only:
            totF = self.myNet.FisherMatr(events)
            totCov, inversion_err = CovMatr(totF)
    
            if self.filt_err != None:
              filtered_err = inversion_err < self.filt_err
              print(np.sum(~filtered_err), 'out of', len(snrs),'dropped by inversion error:', np.sum(filtered_err), 'passed.')
            else:
              filtered_err = np.full(len(snrs), True)

            ####### PASS OR NOT COLUMN
            df['PASS'] = filtered_snr
            df.loc[filtered_snr,'PASS'] = filtered_err
        
            ####### SAVE
            ParNums = IMRPhenomD().ParNums #IMRPhenomD_NRTidalv2
            skyArea = compute_localization_region(totCov, ParNums, events['theta'],units = 'Sterad')
    
            df.loc[filtered_snr,'theta']          = events['theta']
            df.loc[filtered_snr,'phi']            = events['phi']
            df.loc[filtered_snr,'snrs']           = snrs
            df.loc[filtered_snr,'inversion_err']  = inversion_err
            df.loc[filtered_snr,'skyArea']        = skyArea
    
            for key in ['Mc', 'eta', 'dL', 'theta', 'phi']:
              df.loc[filtered_snr,'sigma_'+key] = np.sqrt(np.diagonal(totCov))[:,ParNums[key]]

            return df[self.new_cols]
        
        
        else:
            ####### PASS OR NOT COLUMN
            df['PASS'] = filtered_snr
            # df.loc[filtered_snr,'PASS'] = filtered_err
            
            ####### SAVE
            ParNums = IMRPhenomD().ParNums #IMRPhenomD_NRTidalv2
            # skyArea = compute_localization_region(totCov, ParNums, events['theta'],units = 'Sterad')
    
            # df.loc[filtered_snr,'theta']          = events['theta']
            # df.loc[filtered_snr,'phi']            = events['phi']
            df.loc[filtered_snr,'snrs']           = snrs
            # df.loc[filtered_snr,'inversion_err']  = inversion_err
            # df.loc[filtered_snr,'skyArea']        = skyArea
    
            # for key in ['Mc', 'eta', 'dL', 'theta', 'phi']:
            #   df.loc[filtered_snr,'sigma_'+key] = np.sqrt(np.diagonal(totCov))[:,ParNums[key]]

            return df[self.new_cols_snr_only]
            



    def simulate(self, nevents = None, snr_only = False):
        if nevents == None:
            nevents = len(self.host_g)
        """Splits host_g into batches and processes each using simulate_batch()."""
        
        batch_size = self.batch

        for start in range(0, nevents, batch_size):
            end = min(start + batch_size, nevents)
            batch_indices = self.host_g.index[start:end]

            print(f"Processing batch {start} to {end} (out of {nevents})")
            if not snr_only:
                temp_df = self.simulate_batch(self.host_g.loc[batch_indices].copy(),snr_only = snr_only)  # Call simulate for each batch
                self.host_g.loc[batch_indices,self.new_cols] = temp_df
            else:
                temp_df = self.simulate_batch(self.host_g.loc[batch_indices].copy(),snr_only = snr_only)  # Call simulate for each batch
                self.host_g.loc[batch_indices, self.new_cols_snr_only] = temp_df
            
        print("All batches processed successfully!")


        return None
