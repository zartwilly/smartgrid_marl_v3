#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:27:26 2021

@author: willy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 10:12:26 2021

@author: willy
"""
import os
import time
import pandas as pd
import numpy as np

import fonctions_auxiliaires as fct_aux

K_STEPS_MAX = 50000
NB_INSTANCES = 30

def resume_stat_N_instances(df, path_2_dir_ROOT):
    ids = df[df.C4 == K_STEPS_MAX-1].index
    df.loc[ids, "C2"] = False
    
    ids = df[~((df.C1 == True) & (df.C2 == True))].index
    df.loc[ids, "C3"] = None
    
    # % de OUI dans C1
    percent_OUI_C1 = round(df[df.C1 == True].C1.count()/NB_INSTANCES, 3)
    
    # % de OUI dans C2
    percent_OUI_C2 = round(df[df.C2 == True].C2.count()/NB_INSTANCES, 3)
    
    # % de OUI dans C3
    s_C3 = df.C3.dropna()
    percent_OUI_C3 = None
    if s_C3.size == 0:
        percent_OUI_C3 = 0
    else:
        percent_OUI_C3 = round(s_C3[s_C3 == 1.0].count()/s_C3.count(), 3)
        
    
    # C4: moy du nombre d'etapes stabilisées dans C2
    mean_C4 = df[(df.C2 == True)].C4.mean()
    if np.isnan(mean_C4):
        mean_C4 = 0
    
    # moy des Perfs de RF
    mean_perf_C5 = df.C5.mean()
    
    # moy des perfs de BF
    mean_perf_C6 = df.C6.mean()
    
    # moy de ceux ayant un equilibre de Nash cad C1 = True
    mean_Perf_C7 = df[df.C1 == True].C7.mean()
    
    dico = {"percent_OUI_C1": [percent_OUI_C1], 
            "percent_OUI_C2": [percent_OUI_C2],
            "percent_OUI_C3": [percent_OUI_C3],
            "mean_C4": [mean_C4],
            "mean_perf_C5": [mean_perf_C5],
            "mean_perf_C6": [mean_perf_C6], 
            "mean_Perf_C7": [mean_Perf_C7]
            }
    
    df_res = pd.DataFrame(dico).T
    df_res.columns = ["value"]
    
    df_res.to_excel(os.path.join(path_2_dir_ROOT, 
                                 "resume_stat_50_instances.xlsx"))
    
    df.to_excel(os.path.join(path_2_dir_ROOT, 
                             "RESUME_50_INSTANCES.xlsx"))

if __name__ == "__main__":
    ti = time.time()
    
    phi_name = "A1B1"
    learning_rate = 0.01; ksteps = 25 # 10000 : valeur dans run_game_ATMATE_MULTI_PROCESS_1t_N_instances
    gamma_version = -2; kstoplearn = fct_aux.STOP_LEARNING_PROBA
    name_rep = phi_name+"OnePeriod_50instances_ksteps"\
                        +str(ksteps)\
                        +"_b"+str(learning_rate)\
                        +"_kstoplearn"+str(kstoplearn)
                        
    path_2_dir_ROOT = os.path.join("tests", name_rep)
    file_2_save_all_instances = "save_all_instances"
    path_2_dir = os.path.join(path_2_dir_ROOT,
                              file_2_save_all_instances)
    files_csv = os.listdir(path_2_dir)
    
    df = pd.DataFrame(columns=["C1","C2","C3","C4","C5","C6","C7"])
    for file_csv in files_csv:
        df_tmp = pd.read_csv(os.path.join(path_2_dir, file_csv), index_col=0)
        df = pd.concat([df, df_tmp])
    
    resume_stat_N_instances(df, path_2_dir_ROOT)
    
    print("runtime : {}".format(time.time() - ti))