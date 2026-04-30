import ROOT
import uproot 
import awkward as ak

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

