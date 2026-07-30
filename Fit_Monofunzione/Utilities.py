import ROOT
import uproot 
import awkward as ak
from scipy.stats import crystalball
import numpy as np

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
    return crystalball.pdf(x, beta, m, loc=mu, scale=sigma  )

def extract_and_filter_mass(file_path, tree_name):
    # 1. OTTIMIZZAZIONE I/O: Apriamo il file una sola volta usando 'with' 
    # (così si chiude in automatico liberando memoria)
    with uproot.open(f"{file_path}:{tree_name}") as tree:
        
        # 2. OTTIMIZZAZIONE LETTURA: Leggiamo TUTTI i branch necessari in una sola chiamata
        branches = [
            "FatJet_particleNet_mass",
            "FatJet_particleNetMD_Xbb",
            "FatJet_particleNetMD_QCD",
            "FatJet_isMatchedWithA",
            "FatJet_isMatchedWith2BHadrons"
        ]
        events = tree.arrays(branches, library="ak")
        
    # 3. MASCHERA PARTICLENET
    # Calcoliamo il rapporto. Awkward gestisce automaticamente l'operazione su tutti i jet.
    denominator = events["FatJet_particleNetMD_QCD"] + events["FatJet_particleNetMD_Xbb"]
    tagger_mask = (events["FatJet_particleNetMD_Xbb"] / denominator) >= 0.98
    
    # 4. MASCHERA TRUTH-MATCHING
    # Combiniamo i due controlli booleani
    match_mask = (events["FatJet_isMatchedWithA"] == 1) & (events["FatJet_isMatchedWith2BHadrons"] == 1)
    
    # 5. COMBINAZIONE MASCHERE
    # Un jet deve superare SIA il taglio del tagger SIA il truth matching
    total_mask = tagger_mask & match_mask
    
    # 6. APPLICAZIONE DEL FILTRO E APPIATTIMENTO
    # Applichiamo la maschera alla massa per scartare i jet che non passano i tagli
    filtered_mass = events["FatJet_particleNet_mass"][total_mask]
    
    # Appiattiamo l'array (da struttura annidata 'eventi -> jet' a una singola lista di masse)
    flat_mass = ak.flatten(filtered_mass)
    
    # 7. ORDINAMENTO
    # Convertiamo in un array NumPy standard (più efficiente e compatibile con matplotlib) 
    # e lo ordiniamo dal valore più piccolo al più grande
    sorted_mass = np.sort(ak.to_numpy(flat_mass))
    
    return sorted_mass