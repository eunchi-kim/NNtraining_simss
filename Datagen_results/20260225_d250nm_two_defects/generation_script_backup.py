from joblib import Parallel, delayed
from scipy.stats.qmc import Sobol
from simss_utils.JV_steady_state import *
from copy import deepcopy
import h5py
import glob
import sys

cwd = os.getcwd()
start_time = time.time()

def run(ID, cmd_pars,simss_device_parameters,result_path,session_path=os.path.join(os.getcwd(), 'SIMsalabim','SimSS')):
    try:
    # cwd = os.getcwd()
    # session_path = os.path.join(os.getcwd(), 'SIMsalabim','SimSS')
        G_fracs = None
        JV_file_name = os.path.join(session_path,'JV.dat')
        res = run_SS_JV(simss_device_parameters,result_path,session_path,JV_file_name,G_fracs,parallel=True,force_multithreading=True,cmd_pars=cmd_pars, UUID=ID)
        
    except Exception as e:
        print(f"CRASH at ID {ID}: Parameters: {cmd_pars}")
        print(f"Error Details: {e}")
        return None # Or a failed data flag
    
# Define the function to create the traps file
def create_traps_file_density(folder_path, Et_depth1, Nt1, Nt2, index):
    """
    Creates a new file named traps{index}.txt with the updated second E value.
    """
    new_filename = f"traps{index}.txt"
    full_path = os.path.join(folder_path, new_filename)
    
    content = [
        "E\tNtrap",           # Header
        f"{Et_depth1}\t{Nt1}",         
        f"4.8\t{Nt2}" 
    ]
    
    with open(full_path, 'w') as f:
        f.write("\n".join(content))
    
    return new_filename

# Define the parameters
############################################################################
num_samples = 45000  # Number of samples                                    
varied_parameters = {                                                       
    'W_L': (4.2, 4.5),  # Left defect position range                       
    'W_R': (4.2, 4.5),  # Right defect position range                      
    'E_d,sh': (4.22, 4.48),  # Defect energy range                          
    'N_t,sh': (1e22, 1e24),  # Defect density range                         
    'N_t,d': (1e20, 1e22),  # Defect density range                          
    'mu_e': (1e-8, 1e-6)    # electron mobility                             
}                                                                           
log_transform_items = ['N_t,sh', 'N_t,d', "mu_e"]                           
log_indices = [3,4,5]                                                       
############################################################################
varied_parameters_log = deepcopy(varied_parameters)
for key in log_transform_items:
    if key in varied_parameters_log:
        varied_parameters_log[key] = tuple(np.log(val) for val in varied_parameters_log[key])
        
        
############################################################################
simss_device_parameters = os.path.join(cwd, 'SIMsalabim','SimSS','simulation_setup_sclc_notl.txt')     

# Create the results directory if it doesn't exist
dataset_name = "20260225_two_defects"
res_dir = os.path.join(cwd, "results", dataset_name)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
current_script = sys.argv[0]
shutil.copy2(current_script, os.path.join(res_dir, 'generation_script_backup.py'))
shutil.copy2(simss_device_parameters, os.path.join(res_dir, 'simulation_setup_sclc_notl.txt'))
shutil.copy2(os.path.join(cwd, 'SIMsalabim','SimSS','L2_parameters_OPV.txt'), os.path.join(res_dir, 'L2_parameters_OPV.txt'))
#############################################################################

# Generate the Sobol sequence
sobol = Sobol(d=len(varied_parameters), scramble=True, seed=42)
samples = sobol.random(n=num_samples)

# Scale the samples to the varied parameter ranges
scaled_samples = []
for i, (key, (lower, upper)) in enumerate(varied_parameters_log.items()):
    scaled_samples.append(lower + (upper - lower) * samples[:, i])
scaled_samples = np.array(scaled_samples).T
original_scale_samples = np.copy(scaled_samples)
for idx in log_indices:
    original_scale_samples[:, idx] = np.exp(original_scale_samples[:, idx])

chunk_size = 500
num_batches = int(np.ceil(num_samples / chunk_size))
for bb in range(num_batches):
    start_idx = bb * chunk_size
    end_idx = min((bb + 1) * chunk_size, num_samples)

    print(f"\nProcessing Batch {bb+1}/{num_batches} (Samples {start_idx} to {end_idx})")
    
    batch_samples = original_scale_samples[start_idx:end_idx,:]
    # Generate the dataset
    cmd_pars_list, ID_list = [], []
    for index, sample in enumerate(batch_samples):
        W_L, W_R, E_d_sh, N_t_sh, N_t_d, mu_e = sample
        layer = "l1"
        multitrap_filename = create_traps_file_density(os.path.join(cwd, 'SIMsalabim', 'SimSS'), E_d_sh, N_t_sh, N_t_d,index)
        cmd_pars = [{'par': 'W_L', 'val':str(W_L)}, {'par': 'W_R', 'val':str(W_R)},
                    {'par':'l1.bulkTrapFile', 'val':str(multitrap_filename)},
                    {'par': 'l1.mu_n', 'val': str(mu_e)}]
        cmd_pars_list.append(cmd_pars)
        ID_list.append(str(uuid.uuid4()))
        
    columns = ['ID', 'W_L', 'W_R', 'E_d_sh', 'N_t_sh', 'N_t_d', 'mu_e']
    df_params = pd.DataFrame(batch_samples, columns=columns[1:])
    df_params.insert(0, 'ID', ID_list)
    df_params.to_csv(os.path.join(res_dir, f'parameters_ID_list{bb}.csv'), index=False)

        # Run the simulations in parallel
    Parallel(n_jobs=min(len(cmd_pars_list),10))(delayed(run)(ID, cmd_pars, simss_device_parameters, res_dir) for ID, cmd_pars in zip(ID_list, cmd_pars_list))    

    # Filter out the 'None' entries where simulations failed
    exp_JV_filename = os.path.join(cwd, 'SIMsalabim', 'SimSS', 'exp_JV.csv')
    exp_JV_len = np.loadtxt(exp_JV_filename, skiprows=1).shape[0]
    valid_data = []
    valid_samples = []
    for ID, sample in zip(ID_list, batch_samples):
        JV_file_name = os.path.join(res_dir,f'JV_{ID}.dat')
        log_file_name = os.path.join(res_dir,f'log_{ID}.txt')
        try:
            data = pd.read_csv(JV_file_name,sep=r'\s+')
            data_JV = data['Jext']
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            data_JV = None
        if os.path.exists(JV_file_name):
            os.remove(JV_file_name)
        if os.path.exists(log_file_name):
            os.remove(log_file_name)
        if data_JV is not None:
            if len(data_JV) == exp_JV_len:
                valid_data.append(data_JV) # A/m2
                valid_samples.append(sample)

    # Convert to final arrays
    final_samples = np.array(valid_samples)
    final_jv = np.array(valid_data)
    h5py_filename = os.path.join(res_dir, f'SCLC_Dataset{bb}.h5')
    with h5py.File(h5py_filename, 'w') as hf:
        # Store the input parameters (X)
        hf.create_dataset('inputs', data=final_samples)
        # Store the JV curves (Y)
        hf.create_dataset('outputs', data=final_jv)
        # Store the parameter names for reference
        hf.attrs['feature_names'] = list(varied_parameters.keys())
        # Store the parameter boundaries
        param_group = hf.create_group('config/varied_parameters')
        for key, bounds in varied_parameters.items():
            param_group.attrs[key] = bounds # Saves (min, max)
        log_group = hf.create_group('config/varied_parameters_log')
        for key, bounds in varied_parameters_log.items():
            log_group.attrs[key] = bounds

    print(f"Saved {len(final_samples)} converged samples.")

    traps_files = glob.glob(os.path.join(cwd, 'SIMsalabim', 'SimSS', 'traps*.txt'))
    for file_path in traps_files:
        os.remove(file_path)
    
    del valid_data, valid_samples, final_jv, final_samples
    
end_time = time.time()
total_seconds = end_time - start_time
print(f"Total Execution Time: {total_seconds}s")