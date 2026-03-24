import os
import re
import sys
import uuid
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import optuna

# Ensure 'simss_utils' is in your PYTHONPATH
from simss_utils.JV_steady_state import *
import simss_utils.plot_settings_screen

# ==========================================
# 1. CONFIGURATION & LOGGING
# ==========================================
class Config:
    BASE_DIR = Path.cwd()
    
    # NN / SCLC Paths
    NN_DIR = BASE_DIR / "NN_results" / "20260226-121824_d250nm_two_defects"
    MODEL_PATH = NN_DIR / "y1" / "model.keras"
    SCALER_PATH = NN_DIR / "20260226-121824_d250nm_two_defects_scaler.joblib"
    DATASET_NAME = '20260225_d250nm_two_defects'
    JSON_PATH = BASE_DIR / 'Datagen_results' / DATASET_NAME / 'simulation_metadata.json'
    EXP_SCLC_PATH = BASE_DIR / 'expData' / '2025-06-05-SOP18n7_JVd_#4_1.dat'

    # SIMSS / JVi Paths
    EXP_JVI_PATH = BASE_DIR / "expData" / "2024-01-04-B10n1 LowLight_JVi_#1_4.dat"
    SESSION_PATH = str(BASE_DIR / "SIMsalabim" / "SimSS")
    DEVICE_PARAMS = str(BASE_DIR / "SIMsalabim" / "SimSS" / "simulation_setup.txt")
    
    # Output Dir
    RES_DIR = BASE_DIR / "hybrid_nsga_d328_exp4"
    RES_DIR.mkdir(parents=True, exist_ok=True)

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(Config.RES_DIR / "hybrid_opt.log"), logging.StreamHandler(sys.stdout)])
    return logging.getLogger(__name__)

# ==========================================
# 2. UTILITY & PREPROCESSING FUNCTIONS
# ==========================================
def preprocess_for_nn(v_sim, j_sim, target_v_start, target_v_end, target_points=128):
    v_fwd_target = np.logspace(np.log10(target_v_start), np.log10(target_v_end), target_points)
    v_rev_target = -v_fwd_target
    interp_func = interp1d(v_sim, j_sim, kind='linear', fill_value="extrapolate")
    j_combined = np.concatenate([interp_func(v_fwd_target), interp_func(v_rev_target)])
    return np.log(np.clip(np.abs(j_combined), 1e-10, None))

def preprocess_for_fit(v_sim, j_sim, target_v_start, target_v_end, step=0.02):
    v_target = np.arange(target_v_start, target_v_end + step, step)
    interp_func = interp1d(v_sim, j_sim, kind='linear', fill_value="extrapolate")
    return interp_func(v_target)

def generate_trap_file(output_path, E_d_sh, N_t_sh, N_t_d):
    with open(output_path, 'w') as f:
        f.write(f"E\tNtrap\n{E_d_sh}\t{N_t_sh}\n4.8\t{N_t_d}")

# ==========================================
# 3. OPTIMIZATION OBJECTIVE
# ==========================================
def objective(trial, regressor, param_scaler, metadata, exp_sclc_log, exp_jvi_interp, v_mask_jvi, j_shunt_jvi):
    ID = str(uuid.uuid4())
    jv_temp = Config.RES_DIR / f"JV_{ID}.dat"
    log_temp = Config.RES_DIR / f"log_{ID}.txt"
    trap_temp = Config.RES_DIR / f"traps_{ID}.txt"

    # --- A. Suggest 10D Parameters ---
    # 1. SCLC Parameters 
    lb_mod = list(metadata['varied_parameters_log'].values())       # log transformed boundary
    
    # *Replace these keys with your actual metadata dictionary keys*
    d_sclc = trial.suggest_float("d_sclc", lb_mod[0][0], lb_mod[0][1])
    W_L = trial.suggest_float("W_L", lb_mod[1][0], lb_mod[1][1])
    W_R_sclc = trial.suggest_float("W_R_sclc", lb_mod[2][0], lb_mod[2][1])
    mu_e = trial.suggest_float("mu_e", lb_mod[6][0], lb_mod[6][1])
    E_d_sh = trial.suggest_float("E_d_sh", lb_mod[3][0], lb_mod[3][1])
    N_t_sh = trial.suggest_float("N_t_sh", lb_mod[4][0], lb_mod[4][1])
    N_t_d = trial.suggest_float("N_t_d", lb_mod[5][0], lb_mod[5][1])

    # 2. JVi-specific Parameters
    W_R = trial.suggest_float("W_R", 5.1, 5.4)
    G_frac = trial.suggest_float("G_frac", 0.7, 1.0)
    l1_k_dir = trial.suggest_float("l1_k_dir", 1e-18, 1e-14, log=True)
    l1_C_n_bulk = trial.suggest_float("l1_C_n_bulk", 1e-18, 1e-14, log=True)
    l1_mu_p = trial.suggest_float("l1_mu_p", 1e-8, 1e-6, log=True)

    # --- B. Evaluate SCLC (Neural Network) ---
    # Construct input vector for NN (ensure order matches training!)
    j_min, j_max = metadata['j_min'], metadata['j_max']
    nn_input_raw = np.array([[d_sclc, W_L, W_R_sclc, E_d_sh, N_t_sh, N_t_d, mu_e]]) 
    nn_input_scaled = param_scaler.transform(nn_input_raw)
    
    j_sclc_pred = regressor.predict(nn_input_scaled, verbose=0)[0]
    j_sclc_pred_log = j_sclc_pred_log*(j_max - j_min) + j_min
    
    # Calculate SCLC Loss (e.g., Mean Absolute Error in Log Space)
    sclc_error = float(np.mean(np.abs(j_sclc_pred - exp_sclc_log)))

    # *Crucial optimization:* If SCLC fit is terrible, don't waste time on SIMSS
    if sclc_error > 1.0: # Set this threshold based on your typical NN errors
        raise optuna.TrialPruned()

    # --- C. Evaluate JVi (SIMsalabim) ---
    # Generate temporary trap file for this trial
    generate_trap_file(trap_temp, E_d_sh, np.exp(N_t_sh), np.exp(N_t_d)) # Assuming NN uses log values for traps

    cmd_pars = [
        {'par': 'l1.L', 'val': str(130e-9)},
        {'par': 'W_L', 'val': str(W_L)},
        {'par': 'l1.bulkTrapFile', 'val': str(trap_temp)},
        {'par': 'l1.mu_n', 'val': str(np.exp(mu_e))}, # Convert back from log if necessary
        {'par': 'dev_par_file', 'val': Config.DEVICE_PARAMS},
        {'par': 'JVFile', 'val': str(jv_temp)},
        {'par': 'logFile', 'val': str(log_temp)},
        {'par': 'W_R', 'val': str(W_R)},
        {'par': 'G_frac', 'val': str(G_frac)},
        {'par': 'l1.k_direct', 'val': str(l1_k_dir)},
        {'par': 'l1.C_n_bulk', 'val': str(l1_C_n_bulk)},
        {'par': 'l1.mu_p', 'val': str(l1_mu_p)},
    ]

    try:
        utils_gen.run_simulation('simss', cmd_pars, Config.SESSION_PATH, True, verbose=False)
        data = pd.read_csv(jv_temp, sep=r'\s+')
        
        # Calculate JVi Error (Linear Space MAE)
        jvi_error = float(np.mean(np.abs(data['Jext'][v_mask_jvi] + j_shunt_jvi - exp_jvi_interp[v_mask_jvi])))
        
    except Exception as e:
        logger.warning(f"SimSS failed: {e}")
        jvi_error = 999.0 # Penalty
    finally:
        # Cleanup temp files
        for f in [jv_temp, log_temp, trap_temp]:
            if f.exists(): f.unlink()

    # Return BOTH errors for Multi-Objective optimization
    return sclc_error, jvi_error

# ==========================================
# 4. MAIN WORKFLOW
# ==========================================
def main():
    logger = setup_logging()
    logger.info("Import best-fit parameters from SCLC fitting...")
    # --- Load Initial Parameters from previous step ---
    param_file = Config.RES_DIR / 'best_fit_parameters.csv'
    df_params = pd.read_csv(param_file)
    num_samples = len(list(Config.RES_DIR.glob('fit*.png')))
    
    # Store global parameters for the objective function
    params_row = df_params.iloc[0, num_samples:].values
    global GLOBAL_PARAMS
    GLOBAL_PARAMS = {
        'l1_d' : 130e-9,
        'W_L': params_row[0],
        'mu_e': np.log(params_row[5]),
        'E_d_sh': params_row[2],
        'N_t_sh': np.log(params_row[3]),
        'N_t_d': np.log(params_row[4]),
        'Rpdark': [10.8, 34.6]  # Ohm m2, first and second term of dark shunt resistance
    }
    logger.info(GLOBAL_PARAMS)
    
    
    logger.info("Initializing Hybrid Optuna Workflow...")
    # Load NN Models and Meta
    regressor = load_model(Config.MODEL_PATH)
    param_scaler = joblib.load(Config.SCALER_PATH)
    with open(Config.JSON_PATH, 'r') as f:
        metadata = json.load(f)

    # Prepare SCLC Data
    exp_sclc_data = np.loadtxt(Config.EXP_SCLC_PATH, comments='#')
    exp_sclc_log = preprocess_for_nn(exp_sclc_data[:,0], exp_sclc_data[:,1], 
                                     metadata['target_v_start'], metadata['target_v_end'])

    # Prepare JVi Data
    exp_jvi_data = np.loadtxt(Config.EXP_JVI_PATH, comments='#')
    target_v_start, target_v_end = 0, 1
    exp_jvi_interp = preprocess_for_fit(exp_jvi_data[:,0], exp_jvi_data[:,1]*10*0.16/0.06, target_v_start, target_v_end) # *0.16/0.06 specific for B10 samples
    
    v_target_jvi = np.arange(target_v_start, target_v_end+0.02, 0.02)
    v_mask_jvi = exp_jvi_interp < 0
    Rpdark = [10.8, 34.6]   # specific for each sample
    j_shunt_jvi = v_target_jvi[v_mask_jvi]/Rpdark[0] + v_target_jvi[v_mask_jvi]**2/Rpdark[1]

    # Initialize Optuna Study (Minimize BOTH objectives)
    study = optuna.create_study(directions=["minimize", "minimize"], 
                                study_name="SCLC_JVi_Hybrid",
                                sampler=optuna.samplers.NSGAIISampler()) # NSGA-II is great for multi-objective

    # --- INJECT STARTING POINT ---
    # Here you input the parameters you got from your SCLC-only pre-fit
    best_sclc_params = {
        "W_L": GLOBAL_PARAMS['W_L'], "mu_e": GLOBAL_PARAMS['mu_e'], "E_d_sh": GLOBAL_PARAMS['E_d_sh'], 
        "N_t_sh": GLOBAL_PARAMS['N_t_sh'], "N_t_d": GLOBAL_PARAMS['N_t_d'], # Replace with actual bests
        "W_R": 5.35, "G_frac": 0.8, "l1_k_dir": 1e-16, "l1_C_n_bulk": 1e-16, "l1_mu_p": 1e-7 # Educated guesses
    }
    study.enqueue_trial(best_sclc_params)
    logger.info("Enqueued SCLC pre-fit as starting point.")

    # Run Optimization
    logger.info("Starting Multi-Objective Optimization...")
    # Wrap objective with fixed arguments
    study.optimize(lambda t: objective(t, regressor, param_scaler, metadata, exp_sclc_log, exp_jvi_interp, v_mask_jvi, j_shunt_jvi), 
                   n_trials=100, 
                   n_jobs=1) # Set n_jobs>1 if you want to run SIMSS instances in parallel

    # Extract Pareto Front
    logger.info("Optimization complete. Best trade-off trials:")
    pareto_front = study.best_trials
    for t in pareto_front:
        logger.info(f"Trial {t.number}: SCLC Error = {t.values[0]:.4f}, JVi Error = {t.values[1]:.4f}")

    # Save Study
    df_trials = study.trials_dataframe()
    df_trials.to_csv(Config.RES_DIR / 'optuna_history.csv', index=False)

if __name__ == "__main__":
    main()