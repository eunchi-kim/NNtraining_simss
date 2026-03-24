import re
import sys
import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Optimization imports
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# Ensure 'simss_utils' is in your PYTHONPATH
from simss_utils.JV_steady_state import *
import simss_utils.plot_settings_screen

# ==========================================
# LOGGING & CONFIGURATION
# ==========================================
class Config:
    BASE_DIR = Path.cwd()
    EXP_FOLDER = "20260306-090915_two_defects_7p"
    REG_PATH = BASE_DIR / "NN_results" / EXP_FOLDER
    RES_DIR = REG_PATH / "SE04_2nd_round_fit"
    EXP_PATH = BASE_DIR / "expData" / "2024-01-04-B10n1 LowLight_JVi_#1_11.dat"
    LED_PATH = BASE_DIR / "expData" / 'LED_list.txt'
    
    fit_res_dir = RES_DIR / "fitJVi_exp11"
    fit_res_dir.mkdir(parents=True, exist_ok=True)
    
    # Simulation paths (Update these to your local SIMsalabim environment)
    SESSION_PATH = str(BASE_DIR  / "SIMsalabim" / "SimSS" )
    DEVICE_PARAMS = str(BASE_DIR  / "SIMsalabim" / "SimSS" / "simulation_setup.txt")
    TRAP_FILE = str(BASE_DIR / "SIMsalabim" / "SimSS" / "traps0.txt")
    

def setup_logging(log_file="optimization.log"):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

logger = setup_logging(Config.fit_res_dir / "jvi_fitting.log" )


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def get_sample_name(path: str) -> str:
    """
    Extracts sample name from path using a regex pattern.
    Example: 04-SOP1_..._#1 -> SOP1-1
    """
    filename = os.path.basename(str(path))
    match = re.search(r'([A-Z]\d+n\d+).*?#(\d+)', filename)
    if match:
        sample_base = match.group(1) # B10n1
        pixel_num = match.group(2)   # 1
        sample_id = f"{sample_base}-{pixel_num}"
        return sample_id
    return filename

def preprocess_for_fit(v_sim, j_sim, target_v_start, target_v_end, step=0.02):
    """
    Interpolates simulation data onto a standardized voltage grid.
    """
    v_target = np.arange(target_v_start, target_v_end + step, step)
    interp_func = interp1d(v_sim, j_sim, kind='linear', fill_value="extrapolate")
    return interp_func(v_target)

def get_jv_characteristics(voltage, current_density):
    """
    Calculates Jsc, Voc, FF, and MPP from JV data.
    """
    V = np.array(voltage)
    J = np.array(current_density)
    f_jsc = interp1d(V, J, kind='cubic')
    jsc = abs(float(f_jsc(0)))
    f_voc = interp1d(J, V, kind='cubic')
    voc = float(f_voc(0))
    power = V * J
    idx_mpp = np.argmin(power) 
    p_max = abs(power[idx_mpp])

    ff = p_max / (jsc * voc) * 100
    return {
        'Jsc': jsc/10,'Voc': voc,
        'FF': ff,'MPP': p_max/10}        # convert A/m2 to mA/cm2
# ==========================================
# ANALYSIS & PLOTTING
# ==========================================

def generate_trap_file(output_path, E_d_sh, N_t_sh, N_t_d):
    """Writes the optimized trap parameters to a SIMsalabim-compatible text file."""
    try:
        with open(output_path, 'w') as f:
            f.write(f"E\tNtrap\n{E_d_sh}\t{N_t_sh}\n4.8\t{N_t_d}")
        logger.info(f"Successfully created trap file: {output_path}")
    except Exception as e:
        logger.error(f"Failed to write trap file: {e}")
        
def plot_fitting_results(v_data, j_data, v_fit, j_fit, sample_id, save_path):
    """
    Generates and saves a plot comparing experimental data with the fit.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(v_data, j_data, 'ko', label='Experimental Data')
    j_fit_shunt = v_fit/GLOBAL_PARAMS['Rpdark'][0] +  v_fit**2/GLOBAL_PARAMS['Rpdark'][1]
    plt.plot(v_fit, j_fit+j_fit_shunt, 'r-', label='BO Fit')
    plt.xlabel(r'Voltage $\mathit{V}$ (V)')
    plt.ylabel(r'Current Density $\mathit{J}$ (A/m$^2$)')
    plt.legend()
    plt.ylim(min(j_data)*1.2, min(j_data)*(-0.01))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"JV compare plot saved for {sample_id} at {save_path}")
    
def run_sim_LEDs(sample_id, best_params, save_path, rerun=True):
    """
    run simulation over various LED spectrums
    """
    
    LED_list = np.loadtxt(Config.LED_PATH, dtype=str)
    jv_charac = []
    for LED_name in LED_list:
        final_jv_path = Path(save_path) / f'JV_best_fit_{LED_name}.dat'
        log_path = Path(save_path) / f'log_best_fit_{LED_name}.txt'
        final_cmd = [
            {'par':'JVFile','val': str(final_jv_path)},
            {'par':'spectrum','val': f'../Data/LED_{LED_name}_{sample_id}.txt'},
            {'par': 'dev_par_file', 'val': Config.DEVICE_PARAMS},
            {'par': 'l1.L', 'val': str(GLOBAL_PARAMS['l1_d'])},
            {'par': 'W_L', 'val': str(GLOBAL_PARAMS['W_L'])},
            {'par': 'l1.bulkTrapFile', 'val': Config.TRAP_FILE},
            {'par': 'l1.mu_n', 'val': str(GLOBAL_PARAMS['mu_e'])},
            {'par':'W_R', 'val': str(best_params[1]-best_params[0])},
            {'par':'l1.E_v', 'val': str(best_params[1])},
            {'par':'G_frac', 'val': str(best_params[2])},
            {'par':'l1.k_direct', 'val': str(best_params[3])},
            {'par':'l1.C_n_bulk', 'val': str(best_params[4])},
            {'par':'l1.mu_p', 'val': str(best_params[5])},
        ]
        logger.info(f"best-fit simulation with spectrum from Data/LED_{LED_name}_{sample_id}.txt")
        if rerun:
            utils_gen.run_simulation('simss', final_cmd, Config.SESSION_PATH, True, verbose=False)
        
        data = pd.read_csv(final_jv_path,sep=r'\s+')
        j_sim_shunt = data['Vext']/GLOBAL_PARAMS['Rpdark'][0] +  data['Vext']**2/GLOBAL_PARAMS['Rpdark'][1]
        jv_charac.append(get_jv_characteristics(data['Vext'], data['Jext']+j_sim_shunt))
    df_results = pd.DataFrame(jv_charac)
    output_filename = f'JV_characteristics_{sample_id}.csv'
    output_path = os.path.join(save_path, output_filename)
    df_results.to_csv(output_path, index=False)

    return df_results

def plot_phivar(sim_data, sample_id, save_path):
    exp_phivar_path = Path('expData')/ 'phivar'/ f'{sample_id}_results.xlsx'
    exp_phivar = pd.read_excel(exp_phivar_path,skiprows=[1,2])
    
    sim_data['phi_flux'] = exp_phivar['phi flux'].values
    
    # Experimental data
    exp_phivar['Jsc_norm'] = exp_phivar['Jsc'] / exp_phivar['phi flux'] * 0.1
    exp_phivar['Pout_norm'] = (exp_phivar['Jsc'] * exp_phivar['Voc'] * exp_phivar['FF']) / exp_phivar['phi flux'] * 0.1

    # Simulated 
    sim_data['Jsc_norm'] = sim_data['Jsc'] / sim_data['phi_flux'] * 0.1
    sim_data['Pout_norm'] = (sim_data['Jsc'] * sim_data['Voc'] * sim_data['FF']) / sim_data['phi_flux']* 0.1

    # 2. Create the (2,2) Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    metrics = [
        ('Jsc_norm', '$\mathit{J}_{sc}$ / $\Phi_{in}$ (As)'),
        ('Voc', '$\mathit{V}_{oc}$ (V)'),
        ('FF', 'FF'),
        ('Pout_norm', '$\mathit{P}_{out}$ / $\mathit{\Phi}_{in}$ (Ws)')
    ]

    for i, (col, label) in enumerate(metrics):
        # Plot experimental data as scatter
        axes[i].scatter(exp_phivar['phi flux'], exp_phivar[col if col in exp_phivar else col], 
                        color='black', label='Exp data', alpha=0.7)
        
        # Plot simulation results as a line
        axes[i].plot(sim_data['phi_flux'], sim_data[col], 
                    color='red', label='Prediction', linewidth=2)
        
        axes[i].set_ylabel(label)
        axes[i].set_xscale('log') # Intensity/Flux is usually best viewed on log scale
        if i == 0:
            axes[i].legend()
        if i in [1, 3]:
            axes[i].yaxis.tick_right()
            axes[i].yaxis.set_label_position('right')

    axes[2].set_xlabel(r'Flux density $\mathit{\Phi}_{in}$ (m$^{-2}$s$^{-1}$)')
    axes[3].set_xlabel(r'Flux density $\mathit{\Phi}_{in}$ (m$^{-2}$s$^{-1}$)')

    plt.tight_layout()
    plt.savefig(save_path/'phivar_compare.png')
    plt.close()
    logger.info(f"Phi vs. jv charac plot saved for {sample_id} at {save_path}")

# ==========================================
# OPTIMIZATION SETUP
# ==========================================

# Define the search space
space = [
    Real(0, 0.3, name='W_R'),
    Real(5.3, 5.5, name = 'E_v'),
    Real(0.5, 1.0, name='G_frac'),
    Real(1e-18, 1e-14, prior='log-uniform', name='l1_k_dir'),
    Real(1e-18, 1e-14, prior='log-uniform', name='l1_C_n_bulk'),
    Real(1e-8, 1e-6, prior='log-uniform', name='l1_mu_p')
]

history_data = []

@use_named_args(space)
def objective(W_R, E_v, G_frac, l1_k_dir, l1_C_n_bulk, l1_mu_p):
    """
    Objective function for Bayesian Optimization.
    Minimizes the error between simulation and experimental data.
    """
    ID = str(uuid.uuid4())
    jv_temp = Path(Config.RES_DIR) / f"JV_{ID}.dat"
    log_temp = Path(Config.RES_DIR) / f"log_{ID}.txt"

    # Global variables required by the objective (calculated in main)
    # W_L and mu_e are extracted from the initial fit results
    
    cmd_pars = [
        {'par': 'l1.L', 'val': str(GLOBAL_PARAMS['l1_d'])},
        {'par': 'W_L', 'val': str(GLOBAL_PARAMS['W_L'])},
        {'par': 'l1.bulkTrapFile', 'val': Config.TRAP_FILE},
        {'par': 'l1.mu_n', 'val': str(GLOBAL_PARAMS['mu_e'])},
        {'par': 'dev_par_file', 'val': Config.DEVICE_PARAMS},
        {'par': 'JVFile', 'val': str(jv_temp)},
        {'par': 'logFile', 'val': str(log_temp)},
        {'par': 'W_R', 'val': str(E_v-W_R)},
        {'par': 'l1.E_v', 'val': str(E_v)},
        {'par': 'G_frac', 'val': str(G_frac)},
        {'par': 'l1.k_direct', 'val': str(l1_k_dir)},
        {'par': 'l1.C_n_bulk', 'val': str(l1_C_n_bulk)},
        {'par': 'l1.mu_p', 'val': str(l1_mu_p)},
    ]

    try:
        _ = utils_gen.run_simulation('simss', cmd_pars, Config.SESSION_PATH, True, verbose=False)
        
        # Load simulation result and calculate MAE against experimental data
        data = pd.read_csv(jv_temp, sep=r'\s+')
        error = np.mean(np.abs(data['Jext'][VOLTAGE_MASK] + J_SHUNT - EXP_J_INTERP[VOLTAGE_MASK]))
        
        # Cleanup
        if jv_temp.exists(): jv_temp.unlink()
        if log_temp.exists(): log_temp.unlink()

        history_data.append({
            'W_R': W_R, 'E_v':E_v, 'G_frac': G_frac, 'l1_k_dir': l1_k_dir,
            'l1_C_n_bulk': l1_C_n_bulk, 'l1_mu_p': l1_mu_p, 'error': error
        })
        return error
    
    except Exception as e:
        logger.error(f"Simulation failed for {ID}: {e}")
        return 9e9  # High penalty for failed runs

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def main():
    current_script = sys.argv[0]
    shutil.copy2(current_script, Config.fit_res_dir / 'script_backup.py')
    logger.info("Starting Optimization Workflow")
    logger.info(Config)
    logger.info(space)

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
        'mu_e': params_row[5],
        'E_d_sh': params_row[2],
        'N_t_sh': params_row[3],
        'N_t_d': params_row[4],
        'Rpdark': [10.8, 34.6]  # Ohm m2, first and second term of dark shunt resistance
    }
    logger.info(GLOBAL_PARAMS)
    
    # --- Initialize Experimental Data for comparison ---
    global VOLTAGE_MASK, EXP_J_INTERP, V_TARGET, J_SHUNT
    exp_data = np.loadtxt(Config.EXP_PATH, comments = '#')
    sample_id = get_sample_name(Config.EXP_PATH)
    target_v_start, target_v_end = 0, 1
    EXP_J_INTERP = preprocess_for_fit(exp_data[:,0], exp_data[:,1]*10*0.16/0.06, target_v_start, target_v_end) # *0.16/0.06 is specific for B10 samples
    V_TARGET = np.arange(target_v_start, target_v_end+0.02, 0.02)
    VOLTAGE_MASK = EXP_J_INTERP < 0
    J_SHUNT = V_TARGET[VOLTAGE_MASK]/GLOBAL_PARAMS['Rpdark'][0] + V_TARGET[VOLTAGE_MASK]**2/GLOBAL_PARAMS['Rpdark'][1]
    
    logger.info(f"Data imported from {Config.EXP_PATH}")

    # ---  Run Optimization ---
    logger.info("Running Gaussian Process Minimization...")
    res = gp_minimize(
        objective,
        space,
        n_calls=100,
        n_initial_points=10,
        random_state=42
    )

    hist_path = Config.fit_res_dir / 'optimization_history.csv'
    pd.DataFrame(history_data).to_csv(hist_path, index=False)
    logger.info(f"Optimization finished. History saved to {hist_path}")
    logger.info(f"Best parameters: {res.x}")
    
    best_params = res.x
    final_jv_path = os.path.join(Config.fit_res_dir, 'best_fit_JV.dat')
    log_path = os.path.join(Config.fit_res_dir, 'best_fit_log.txt')

    # Map best_params back to your simulation command
    final_cmd = [
        {'par':'JVFile','val':final_jv_path},
        {'par':'logFile','val':log_path},
        {'par': 'l1.L', 'val': str(GLOBAL_PARAMS['l1_d'])},
        {'par': 'W_L', 'val': str(GLOBAL_PARAMS['W_L'])},
        {'par': 'l1.bulkTrapFile', 'val': Config.TRAP_FILE},
        {'par': 'l1.mu_n', 'val': str(GLOBAL_PARAMS['mu_e'])},
        {'par': 'dev_par_file', 'val': Config.DEVICE_PARAMS},
        {'par':'W_R', 'val': str(best_params[1]-best_params[0])},
        {'par':'l1.E_v', 'val': str(best_params[1])},
        {'par':'G_frac', 'val': str(best_params[2])},
        {'par':'l1.k_direct', 'val': str(best_params[3])},
        {'par':'l1.C_n_bulk', 'val': str(best_params[4])},
        {'par':'l1.mu_p', 'val': str(best_params[5])},
    ]

    _ = utils_gen.run_simulation('simss', final_cmd, Config.SESSION_PATH, True, verbose=False)
    best_sim_data = pd.read_csv(final_jv_path, sep=r'\s+')
    plot_path = Config.fit_res_dir / f"fit_compare.png"
    plot_fitting_results(V_TARGET[VOLTAGE_MASK], EXP_J_INTERP[VOLTAGE_MASK], 
                         best_sim_data['Vext'], best_sim_data['Jext'], sample_id, plot_path)
    bestfit_phivar = run_sim_LEDs(sample_id, best_params, Config.fit_res_dir)
    plot_phivar(bestfit_phivar, sample_id, Config.fit_res_dir)

if __name__ == "__main__":
    main()