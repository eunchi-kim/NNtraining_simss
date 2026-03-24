from joblib import Parallel, delayed
from scipy.stats.qmc import Sobol
from simss_utils.JV_steady_state import *
from copy import deepcopy
import h5py
import glob
import sys
import io
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    
def run_simulation_instance(sample, index, sim_setup, res_dir, cwd):
    """Encapsulates a single simulation run to avoid global state issues."""
    l1_L, W_L, W_R, E_d_sh, N_t_sh, mu_e = sample
    ID = str(uuid.uuid4())
    
    # Use a unique trap filename per process to avoid collisions
    trap_fn = os.path.join(cwd, 'SIMsalabim', 'SimSS', f"traps_{index}.txt")
    
    try:
        # Create trap file
        with open(trap_fn, 'w') as f:
            f.write(f"E\tNtrap\n{E_d_sh}\t{N_t_sh}")

        cmd_pars = [
            {'par': 'l1.L', 'val': str(l1_L)},
            {'par': 'W_L', 'val': str(W_L)}, 
            {'par': 'W_R', 'val': str(W_R)},
            {'par': 'l1.bulkTrapFile', 'val': trap_fn},
            {'par': 'l1.mu_n', 'val': str(mu_e)}
        ]
        
        # Run simulation
        run(ID, cmd_pars, sim_setup, res_dir)
        
        # Read and validate immediately
        jv_path = os.path.join(res_dir, f'JV_{ID}.dat')
        log_path = os.path.join(res_dir, f'log_{ID}.txt')
        try:
            data = pd.read_csv(jv_path, sep=r'\s+')
            data_JV = data['Jext']
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            data_JV = None
        
        # Cleanup immediately after reading
        if os.path.exists(jv_path): os.remove(jv_path)
        if os.path.exists(log_path): os.remove(log_path)
        if os.path.exists(trap_fn): os.remove(trap_fn)
        
        return {"ID": ID, "sample": sample, "J": data_JV}
    
    except Exception as e:
        # Clean up trap file even if simulation crashes
        if os.path.exists(trap_fn): os.remove(trap_fn)
        return None

if __name__ == '__main__':
    # Define the parameters
    ############################################################################
    num_samples = 20000  # Number of samples   
    chunk_size = 500                                
    varied_parameters = {      
        'l1_l' : (50e-9, 400e-9),   # active layer thickness                                                 
        'W_L': (4.2, 4.5),  # Left defect position range                       
        'W_R': (4.2, 4.5),  # Right defect position range                      
        'E_d': (4.22, 4.8),  # Defect energy range                          
        'N_t': (1e20, 1e24),  # Defect density range                         
        # 'N_t,d': (1e20, 1e22),  # Defect density range                          
        'mu_e': (1e-8, 1e-6)    # electron mobility                             
    }                                                                           
    log_transform_items = ['N_t', "mu_e"]                           
    log_indices = [4,5]                                                       
    ############################################################################
    varied_parameters_log = deepcopy(varied_parameters)
    for key in log_transform_items:
        if key in varied_parameters_log:
            varied_parameters_log[key] = tuple(np.log(val) for val in varied_parameters_log[key])
            
            
    ############################################################################
    simss_device_parameters = os.path.join(cwd, 'SIMsalabim','SimSS','simulation_setup_sclc_notl.txt')     

    # Create the results directory if it doesn't exist
    dataset_name = "20260323_one_defects_6p"
    res_dir = os.path.join(cwd, "Datagen_results", dataset_name)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    current_script = sys.argv[0]
    shutil.copy2(current_script, os.path.join(res_dir, 'generation_script_backup.py'))
    shutil.copy2(simss_device_parameters, os.path.join(res_dir, 'simulation_setup_sclc_notl.txt'))
    shutil.copy2(os.path.join(cwd, 'SIMsalabim','SimSS','L2_parameters_OPV.txt'), os.path.join(res_dir, 'L2_parameters_OPV.txt'))
    #############################################################################
    exp_JV_filename = os.path.join(cwd, 'SIMsalabim', 'SimSS', 'exp_JV.csv')
    exp_JV_len = np.loadtxt(exp_JV_filename, skiprows=1).shape[0]

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
        
    # --- INITIALIZATION (for data saving) ---
    h5py_filename = os.path.join(res_dir, f'{dataset_name}_Complete.h5')
    num_features = len(varied_parameters)

    with h5py.File(h5py_filename, 'w') as hf:
        hf.create_dataset('inputs', shape=(0, num_features), maxshape=(None, num_features), 
                        chunks=True, compression="gzip")
        hf.create_dataset('outputs', shape=(0, exp_JV_len), maxshape=(None, exp_JV_len), 
                        chunks=True, compression="gzip")
        
        # Store static metadata
        hf.attrs['feature_names'] = list(varied_parameters.keys())
        param_group = hf.create_group('config/varied_parameters')
        for key, bounds in varied_parameters.items():
            param_group.attrs[key] = bounds

    num_batches = int(np.ceil(num_samples / chunk_size))
    for bb in range(num_batches):
        start_idx = bb * chunk_size
        end_idx = min((bb + 1) * chunk_size, num_samples)

        print(f"\nProcessing Batch {bb+1}/{num_batches} (Samples {start_idx} to {end_idx})")
        
        batch_samples = original_scale_samples[start_idx:end_idx,:]
        # Generate the dataset
        valid_data = []
        valid_samples = []
        # Filter out the 'None' entries where simulations failed
        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_simulation_instance, s, i, simss_device_parameters, res_dir, cwd) 
                for i, s in enumerate(batch_samples)]
        
            for future in as_completed(futures):
                result = future.result()
                if result['J'] is not None and len(result['J']) == exp_JV_len:
                    valid_data.append(result['J'])
                    valid_samples.append(result['sample'])

        if len(valid_samples) > 0:
            with h5py.File(h5py_filename, 'a') as hf:
                # Resize datasets to accommodate new data
                curr_size = hf['inputs'].shape[0]
                new_size = curr_size + len(valid_samples)
                
                hf['inputs'].resize((new_size, num_features))
                hf['outputs'].resize((new_size, exp_JV_len))
                
                # Write batch data
                hf['inputs'][curr_size:new_size, :] = np.array(valid_samples)
                hf['outputs'][curr_size:new_size, :] = np.array(valid_data)

        print(f"Saved {len(valid_samples)} converged samples.")
        del valid_data, valid_samples
        
    end_time = time.time()
    total_seconds = end_time - start_time
    print(f"Total Execution Time: {total_seconds}s")