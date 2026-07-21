import uproot 
import awkward as ak
from scipy.stats import crystalball
from scipy.interpolate import interp1d
from scipy.special import voigt_profile
import json
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares


def extractor(file_path, tree_name):
    file=uproot.open(file_path)
    tree=file[tree_name]
    branches=["FatJet_particleNet_mass", "FatJet_particleNetMD_Xbb", "FatJet_particleNetMD_QCD"]
    events=tree.arrays(branches, library="ak")
    Xbb=events["FatJet_particleNetMD_Xbb"]
    QCD=events["FatJet_particleNetMD_QCD"]      
    #filter=FatJet_particleNetMD_Xbb/(FatJet_particleNetMD_QCD+FatJet_particleNetMD_Xbb) >= 0.98
    #Questo è il filtro datoci da Carlo (Chiedere al prof cos'è QCD)
    filtro_25=Xbb/(QCD+Xbb) >= 0.98
    dati_FatJet=tree.arrays("FatJet_particleNet_mass", library="ak", cut=filtro_25)
    #Otteniamo un array piatto col quale possiamo lavorare
    dati_piatti=ak.flatten(dati_FatJet["FatJet_particleNet_mass"])    
    return dati_piatti

def cb_pdf(x, mu, sigma, beta, m):
    return crystalball.pdf(-1*x, beta, m, loc=-mu, scale=sigma  )


#Fit per una massa non data

def voigt2(x, norm, mu, sigma, gamma, norm2, mu2, sigma2, gamma2):
    return voigt_profile(x-mu, sigma, gamma) * norm + norm2*voigt_profile(x-mu2, sigma2, gamma2)


masse=[25, 50, 75, 100, 125, 150, 175, 200]

with open("fit_values.json", "r") as f:
    values = json.load(f)
#Siamo riusciti a passare i dati dei fit al file json, ora possiamo usarli per fare i grafici.

#Interpoliamo le norme

masse=[25, 50, 75, 100, 125, 150, 175, 200]
fit_MH_values=values


class DoubleVoigtFit:
    def __init__(self, mass, fit_values, ax=None, color=None,masse =[25, 50, 75, 100, 125, 150, 175, 200]):
        self.mass = mass
        self.fit_values = fit_values
        self.ax = ax
        self.color = color
        self.masse = masse

    def get_fit_parameters(self):
        # Interpoliamo i parametri per la massa specificata
        norm_interp = interp1d(self.masse, [fit["norm"] for fit in self.fit_values.values()], kind='linear', fill_value='extrapolate')
        norm2_interp = interp1d(self.masse, [fit["norm2"] for fit in self.fit_values.values()], kind='linear', fill_value='extrapolate')
        mu_interp = interp1d(self.masse, [fit["mu"] for fit in self.fit_values.values()], kind='linear', fill_value='extrapolate')
        mu2_interp = interp1d(self.masse, [fit["mu2"] for fit in self.fit_values.values()], kind='linear', fill_value='extrapolate')
        sigma_interp = interp1d(masse, [fit["sigma"] for fit in self.fit_values.values()], kind='linear', fill_value='extrapolate')
        sigma2_interp = interp1d(masse, [fit["sigma2"] for fit in self.fit_values.values()], kind='linear', fill_value='extrapolate')
        gamma_interp = interp1d(masse, [fit["gamma"] for fit in self.fit_values.values()], kind='linear', fill_value='extrapolate')
        gamma2_interp = interp1d(masse, [fit["gamma2"] for fit in self.fit_values.values()], kind='linear', fill_value='extrapolate')

        return {
            "norm": norm_interp(self.mass),
            "norm2": norm2_interp(self.mass),
            "mu": mu_interp(self.mass),
            "mu2": mu2_interp(self.mass),
            "sigma": sigma_interp(self.mass),
            "sigma2": sigma2_interp(self.mass),
            "gamma": gamma_interp(self.mass),
            "gamma2": gamma2_interp(self.mass)
        }
    
    def grafico(self, ax=None, color=None):
        fit_parameters = self.get_fit_parameters()
        x = np.linspace(0, 250, 300)
        y = voigt2(x, fit_parameters["norm"], fit_parameters["mu"], fit_parameters["sigma"], fit_parameters["gamma"], fit_parameters["norm2"], 
                   fit_parameters["mu2"], fit_parameters["sigma2"], fit_parameters["gamma2"])
        
        
        if ax is None:
            plt.figure(figsize=(10, 6))
        else:
            plt.sca(ax)
        plt.plot(x, y, label=f'Interpolatepolazione per le Masse {self.mass}', color=self.color)
        plt.title(f'Interpolazione per la Massa {self.mass}')
        plt.xlabel('Massa')
        plt.ylabel('Densità')
        plt.legend()
        plt.grid()
        
        return
    
class DoubleVoigtFit2:
    def __init__(self, mass, fit_values, exclude_mass=50, ax=None, color=None):
        self.mass = mass
        self.ax = ax
        self.color = color
        
        self.param_names = ["norm", "mu", "sigma", "gamma", "norm2", "mu2", "sigma2", "gamma2"]
        
        exclude_key = f"MH{exclude_mass}"
        cleaned_data = {float(k.replace("MH", "")): params for k, params in fit_values.items() if str(k).startswith("MH") and k != exclude_key}

        self.masse = sorted(cleaned_data.keys())

        self.y_params = np.array([[cleaned_data[m][p] for m in self.masse] for p in self.param_names])

    def get_fit_parameters(self):
        interpolator = interp1d(self.masse, self.y_params, kind='linear', fill_value='extrapolate')
        
        interpolated_values = interpolator(self.mass)
        
        return dict(zip(self.param_names, interpolated_values))
    
    def grafico(self):
        fit_params = self.get_fit_parameters()
        x = np.linspace(0, 250, 300)
        
       
        args_for_voigt = [fit_params[p] for p in self.param_names]
        y = voigt2(x, *args_for_voigt)
        
        ax = self.ax if self.ax is not None else plt.subplots(figsize=(10, 6))[1]
            
        ax.plot(x, y, label=f'Curva Interpolata (Massa {self.mass})', color=self.color)
        ax.set_title(f'Interpolazione per la Massa {self.mass}')
        ax.set_xlabel('Massa')
        ax.set_ylabel('Densità')
        ax.legend()
        ax.grid(True)

