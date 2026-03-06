from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize
import numpy as np


class OptimizationProblem(Problem):
    """
    A pymoo Problem class that wraps the cost evaluation function.
    It handles the scaling of parameters from the optimizer's [0, 1] space
    to the actual problem's parameter space.
    """
    def __init__(self, n_var, lb, ub, cost_function, cost_func_kwargs):
        super().__init__(
            n_var=n_var,
            n_obj=1,
            xl=np.zeros(n_var),
            xu=np.ones(n_var),
        )
        self.lb = lb
        self.ub = ub
        self.cost_function = cost_function
        self.cost_func_kwargs = cost_func_kwargs

    def _evaluate(self, theta_normalized, out, *args, **kwargs):
        """
        Evaluates the cost function for a batch of normalized theta vectors.
        """
        # Scale theta from the optimizer's [0, 1] space to the problem's log-space
        theta_log_space = theta_normalized * (self.ub - self.lb) + self.lb
        
        # Calculate the error using the provided cost function
        error = self.cost_function(theta_log_space, **self.cost_func_kwargs)
        
        out["F"] = error

class PopulationCallback(Callback):
    """
    A simple pymoo callback to store the parameter vectors (offspring)
    of each generation during the optimization process.
    """
    def __init__(self):
        super().__init__()
        # Use a more descriptive name for the stored data
        self.data["generation_offspring"] = []

    def notify(self, algorithm):
        # Store all offspring from the current population
        self.data["generation_offspring"].append(algorithm.pop.get("X"))

# --- Main Function ---

def run_cmaes_sclc(
    n_var: int,
    lb_mod: np.ndarray,
    ub_mod: np.ndarray,
    obs_mask: np.ndarray,
    cost_func_kwargs: dict,
    popsize: int = 200,
    n_gen: int = 100,
    sigma: float = 0.2,
    weight: float = 1,
    joint_opt: bool = False,
    verbose: bool = False
):
    """
    Runs the CMA-ES optimization algorithm to find the best-fit parameters.

    Args:
        n_var (int): The number of variables to optimize.
        lb_mod (np.ndarray): The lower bounds of the parameters (in log-space).
        ub_mod (np.ndarray): The upper bounds of the parameters (in log-space).
        obs_mask (np.ndarray): A boolean mask for the cost function.
        cost_func_kwargs (dict): A dictionary of all arguments needed by the `evaluate_cost` function.
        popsize (int, optional): The population size for the algorithm. Defaults to 200.
        n_gen (int, optional): The number of generations to run. Defaults to 100.
        sigma (float, optional): The initial step size for CMA-ES. Defaults to 0.2.
        verbose (bool, optional): If True, prints execution time and results. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - res.X (np.ndarray): Best parameters in pymoo's normalized [0, 1] space.
            - best_full (np.ndarray): Best parameters reconstructed into the full, log-space vector.
            - best_norm (np.ndarray): Best parameters transformed by the standard scaler.
            - res.F (float): The final error (cost) of the best solution.
            - callback_data (dict): A dictionary containing the history of explored parameters.
    """
    # 1. Define the optimization problem
    # Pass all cost function arguments in a single dictionary for cleanliness
    problem = OptimizationProblem(
        n_var=n_var,
        lb=lb_mod,
        ub=ub_mod,
        cost_function=_calculate_sclc_error,  # Assuming 'evaluate_cost' is your cost function
        cost_func_kwargs={'obs_mask': obs_mask,**cost_func_kwargs}
    )
    
    if joint_opt:
        problem = OptimizationProblem(
        n_var=n_var,
        lb=lb_mod,
        ub=ub_mod,
        cost_function=_calculate_sclc_joint_fit_error_batch,  # Assuming 'evaluate_cost' is your cost function
        cost_func_kwargs={'obs_mask': obs_mask,**cost_func_kwargs}
    )

    # 2. Define the algorithm
    algorithm = CMAES(
        popsize=popsize,
        sigma=sigma,
        restarts=0,
        restart_from_best=False
    )

    # 3. Run the optimization
    callback = PopulationCallback()
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', n_gen),
        callback=callback,
        verbose=False # Internal verbosity is off; we handle it manually
    )

    # 4. Process the results
    # Un-scale the best result from [0, 1] back to the problem's log-space
    best_optimized = res.X * (ub_mod - lb_mod) + lb_mod
    best_full = best_optimized.reshape(1, -1)
    
    # Transform the full vector using the provided scaler
    jvi_scaler = cost_func_kwargs['scalers']['sclc_stscaler']
    if joint_opt:
        # Reconstruct the (N, 7) matrix for the best solution to allow scaling
        num_samples = cost_func_kwargs['data']['y_exp_sclc_log'].shape[0]
        
        best_thicknesses = best_optimized[:num_samples]
        best_shared_material = best_optimized[num_samples:]
        
        # Tile and stack to create the (N, 7) matrix
        best_full = np.column_stack((
            best_thicknesses, 
            np.tile(best_shared_material, (num_samples, 1))
        ))
        
        # best_norm will now be (N, 7)
        best_norm = jvi_scaler.transform(best_full)
    else:
        # Standard 7-param logic
        best_full = best_optimized.reshape(1, -1)
        best_norm = jvi_scaler.transform(best_full)

    if verbose:
        print(f"CMA-ES optimization took {res.exec_time:.2f} seconds.")
        print(f"Best solution error: {res.F[0]:.4f}")
        print(f"Best full parameters (log-space): \n{best_full}")

    callback_data = callback.data
    return res.X, best_optimized, best_full, best_norm, res.F, callback_data



def _calculate_sclc_error(
    theta_sclc: np.ndarray, obs_mask: np.ndarray, models: dict, scalers: dict, data: dict, config: dict
) -> float:
    """Calculates the total RMSE for the SCLC model predictions."""
    # 1. Normalize parameters and make predictions
    theta_norm_sclc = scalers['sclc_stscaler'].transform(theta_sclc)
    y_pred = models['sclc_reg'].predict(theta_norm_sclc, verbose=0)
    y_pred_log = scale_back(y_pred, scalers['y1_min_sclc'], scalers['y1_max_sclc'])

    # 2. Calculate RMSE for each curve segment based on the voltage mask
    errors = []
    num_curves = config['sclc_nn_points'] // config['sclc_one_jv_len']
    for i in range(num_curves):
        start, end = i * config['sclc_one_jv_len'], (i + 1) * config['sclc_one_jv_len']
        
        # Create a mask for the current curve segment using the provided voltage_mask
        segment_mask = np.zeros(config['sclc_nn_points'], dtype=bool)
        segment_mask[start:end] = data['voltage_mask'][start:end]
        
        # Calculate the error for the masked region
        error_diff = y_pred_log[:, segment_mask] - data['y_exp_sclc_log'][segment_mask]
        rmse = np.sqrt(np.mean(error_diff**2, axis=-1))
        errors.append(rmse)
        
    errors = np.vstack(errors).T
    
    # 3. Sum the errors for the observed curves
    total_error = np.sum(errors[:, obs_mask], axis=-1)
    
    return total_error

def _calculate_sclc_joint_fit_error_batch(
    theta_pop: np.ndarray, # Shape: (pop_size, N + 6)
    obs_mask: np.ndarray, 
    models: dict, 
    scalers: dict, 
    data: dict, 
    config: dict
) -> np.ndarray: # Returns array of errors of length (pop_size)
    
    y_exp_all = data['y_exp_sclc_log'] # Shape: (N, points)
    num_samples = y_exp_all.shape[0]
    pop_size = theta_pop.shape[0]
    
    # 1. Split Population
    # thicknesses: (pop_size, N)
    # shared_params: (pop_size, 6)
    thicknesses = theta_pop[:, :num_samples]
    shared_params = theta_pop[:, num_samples:]

    # 2. Reshape for the Neural Network
    # We need to create a row for every (individual, sample) combination
    # Final shape needed: (pop_size * N, 7)
    
    # Repeat each individual's shared params N times
    # shared_expanded shape: (pop_size * N, 6)
    shared_expanded = np.repeat(shared_params, num_samples, axis=0)
    
    # Flatten thicknesses: [ind1_s1, ind1_s2, ind2_s1, ind2_s2...]
    # thick_flat shape: (pop_size * N, 1)
    thick_flat = thicknesses.reshape(-1, 1)
    
    # Combine into the 7-parameter input
    full_input_batch = np.column_stack((thick_flat, shared_expanded))

    # 3. Batch Prediction
    theta_norm = scalers['sclc_stscaler'].transform(full_input_batch)
    y_pred = models['sclc_reg'].predict(theta_norm, verbose=0)
    y_pred_log = scale_back(y_pred, scalers['y1_min_sclc'], scalers['y1_max_sclc'])
    
    # Reshape predictions back to (pop_size, num_samples, points)
    y_pred_reshaped = y_pred_log.reshape(pop_size, num_samples, -1)

    # 4. Vectorized Error Calculation
    # We compare y_pred_reshaped (pop, N, pts) with y_exp_all (N, pts)
    # Calculate squared difference
    diff_sq = (y_pred_reshaped[:,:,data['voltage_mask']] - y_exp_all[:,data['voltage_mask']])**2
    
    # Calculate RMSE per sample, per individual
    # We average over the points dimension (axis=2)
    sample_rmses = np.sqrt(np.mean(diff_sq, axis=2)) # Shape: (pop_size, num_samples)
    
    # 5. Final Summation
    # Sum the errors across all samples for each individual in the population
    total_errors = np.sum(sample_rmses, axis=1) # Shape: (pop_size,)
    
    return total_errors

def scale_and_exponentiate(pred, min_val, max_val):
    return np.exp(pred * (max_val - min_val) + min_val)

def scale_back(pred, min_val, max_val):
    return pred * (max_val - min_val) + min_val

def log_and_standardize(pred, min_val, max_val): 
    return (np.log(pred) - min_val) / (max_val - min_val)