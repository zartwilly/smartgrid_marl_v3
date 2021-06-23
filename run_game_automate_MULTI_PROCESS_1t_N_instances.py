#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:08:21 2021

@author: willy
"""

import os
import time
import numpy as np
import pandas as pd
import multiprocessing as mp
import execution_game_automate_4_all_t as autoExeGame4T
import fonctions_auxiliaires as fct_aux

from pathlib import Path

# _________             define paramters of main : debut            ___________
def define_parameters_MULTI_gammaV_instances_phiname_arrplMTVars(dico_params):
    params = []
    
    for gamma_version in dico_params["gamma_versions"]:
        
        for phi_name, dico_ab in dico_params["dico_phiname_ab"].items():
        
            # ----   execution of 50 instances    ----
            name_dir_oneperiod \
                = os.path.join(
                    dico_params["name_dir"],
                    #"OnePeriod_50instances",
                    phi_name+"OnePeriod_50instances",
                    "OnePeriod_"+str(dico_params["nb_instances"])+"instances"+"GammaV"+str(gamma_version))
            
            for numero_instance in range(0, dico_params["nb_instances"]):
                arr_pl_M_T_vars_init \
                    = fct_aux.get_or_create_instance_Pi_Ci_one_period_doc23(
                        dico_params["setA_m_players"], 
                        dico_params["setB_m_players"], 
                        dico_params["setC_m_players"], 
                        dico_params["t_periods"], 
                        dico_params["scenario"],
                        dico_params["scenario_name"],
                        dico_params["path_to_arr_pl_M_T"], 
                        dico_params["used_instances"])
                    
                date_hhmm_new = "_".join([date_hhmm, str(numero_instance), 
                                          "t", str(dico_params["t_periods"])])
                
                param = [arr_pl_M_T_vars_init, 
                         name_dir_oneperiod,
                         date_hhmm_new,
                         k_steps,
                         dico_params["NB_REPEAT_K_MAX"],
                         dico_params["algos"],
                         dico_params["learning_rates"],
                         dico_params["pi_hp_plus"],
                         dico_params["pi_hp_minus"],
                         dico_ab['a'], 
                         dico_ab['b'],
                         gamma_version,
                         dico_params["used_instances"],
                         dico_params["used_storage_det"],
                         dico_params["manual_debug"], 
                         dico_params["criteria_bf"], 
                         numero_instance,
                         dico_params["debug"] 
                         ]
                
                params.append(param)
                
    return params

# _________             define paramters of main : fin            ___________

if __name__ == "__main__":
    ti = time.time()
    
    # constances 
    criteria_bf = "Perf_t"
    used_storage_det = True #False #True
    manual_debug = False #True #False #True
    used_instances = False #True
    debug = False
    
    date_hhmm="DDMM_HHMM"
    t_periods = 1 #50 #30 #35 #55 #117 #15 #3
    k_steps = 25 #50000 #250 #5000 #2000 #50 #250
    NB_REPEAT_K_MAX= 3 #10 #3 #15 #30
    learning_rates = [0.01]#[0.1] #[0.001]#[0.00001] #[0.01] #[0.0001]
    fct_aux.N_DECIMALS = 8
    dico_phiname_ab = {"A1B1": {"a":1, "b":1}, "A1.2B0.8": {"a":1.2, "b":0.8}}
    dico_phiname_ab = {"A1B1": {"a":1, "b":1}}
    pi_hp_plus = [10] #[10] #[0.2*pow(10,-3)] #[5, 15]
    pi_hp_minus = [30] #[20] #[0.33] #[15, 5]
    fct_aux.PI_0_PLUS_INIT = 4 #20 #4
    fct_aux.PI_0_MINUS_INIT = 3 #10 #3
    NB_INSTANCES = 30 #50
            
    algos = fct_aux.ALGO_NAMES_LRIx \
            + fct_aux.ALGO_NAMES_DET \
            + fct_aux.ALGO_NAMES_BF \
            + fct_aux.ALGO_NAMES_NASH 
    
            
    # ---- initialization of variables for generating instances ----
    setA_m_players, setB_m_players, setC_m_players = 15, 10, 10                # 35 players 
    setA_m_players, setB_m_players, setC_m_players = 10, 6, 5                  # 21 players 
    setA_m_players, setB_m_players, setC_m_players = 8, 4, 4                   # 16 players
    setA_m_players, setB_m_players, setC_m_players = 6, 3, 3                   # 12 players
                      
    scenario_name = "scenarioOnePeriod"
    scenario = None
    
    name_dir = "tests"
    path_to_arr_pl_M_T = os.path.join(*[name_dir, "AUTOMATE_INSTANCES_GAMES"])
    
    gamma_versions = [-2] #-1 : random normal distribution, 0: not stock anticipation, -2: normal distribution with proba ppi_k
    
    dico_params = {"dico_phiname_ab":dico_phiname_ab, 
        "gamma_versions":gamma_versions,
        "nb_instances":NB_INSTANCES,
        "criteria_bf":criteria_bf, "used_storage_det":used_storage_det,
        "manual_debug":manual_debug, "debug":debug,
        "date_hhmm":date_hhmm,
        "pi_hp_plus":pi_hp_plus, "pi_hp_minus":pi_hp_minus,
        "algos":algos,"learning_rates":learning_rates,
        "setA_m_players":setA_m_players, 
        "setB_m_players":setB_m_players, 
        "setC_m_players":setC_m_players,
        "t_periods":t_periods,"scenario":scenario,"scenario_name":scenario_name,
        "path_to_arr_pl_M_T":path_to_arr_pl_M_T,"used_instances":used_instances, 
        "name_dir":name_dir, 
        "NB_REPEAT_K_MAX": NB_REPEAT_K_MAX}
    
    params = define_parameters_MULTI_gammaV_instances_phiname_arrplMTVars(dico_params)
    print("define parameters finished")
    
    # multi processing execution
    p = mp.Pool(mp.cpu_count()-1)
    p.starmap(
        autoExeGame4T.execute_algos_used_Generated_instances_N_INSTANCES_MULTI,
        params
    )
    # multi processing execution
    
    print("Multi process running time ={}".format(time.time()-ti))
    