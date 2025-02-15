"""
Serves as the entry point for conducting experiments. 
It initializes hyperparameters, calls run_experiments(...) 
from experiments.py, and handles any final aggregated output.
"""


import torch

from experiments import run_experiments


def get_eigenvalues_for_condition(d, cond=10.0):
    """
    Generate a list of eigenvalues of length d that yields a condition number of ~cond.
    Here, we simply set the largest eigenvalue to cond and all others to 1.
    """
    eigenvals = [1.0] * d
    eigenvals[0] = cond
    return eigenvals

if __name__ == "__main__":
    # Fixed training hyperparameters
    num_experiments = 3         # number of seeds per configuration
    num_steps = 3000
    metric_interval = 25
    val_batch_size = 2000
    compare_mode = "manual"     # "manual" baseline (PGD)
    noise_variance = 0.1        # noise variance for data generation
    line_search_min = 0.001
    line_search_max = 2.0
    line_search_steps = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameter grids for the experiments:
    B_values = [32]                     # Only one value for training batch size.
    N_values = [20, 100]
    d_values = [5, 20]

    # Define model variants as tuples: (model_label, activation_choice, leaky_alpha)
    model_variants = [
        ("linear",     None,        0.5),   # linear (default)
        ("relu",       "relu",      0.5),   # ReLU (alpha is ignored)
        ("leakyrelu0.75", "leakyrelu", 0.75),
        ("leakyrelu0.5",  "leakyrelu", 0.5),
        ("leakyrelu0.25", "leakyrelu", 0.25)
    ]

    # Data modes: each is a tuple: (data_label, function to obtain eigenvalues_list)
    # For isotropic data, we use None for eigenvalues_list.
    def isotropic(_d):
        return None
    def non_isotropic(d):
        return get_eigenvalues_for_condition(d, cond=10.0)
    data_modes = [
        ("iso", isotropic),
        ("noniso", non_isotropic)
    ]

    total_runs = 0
    # Loop over all combinations.
    for B in B_values:
        for N in N_values:
            for d in d_values:
                for (model_label, activation_choice, leaky_alpha) in model_variants:
                    for (data_label, data_func) in data_modes:
                        total_runs += 1
                        print(f"\n=== Configuration #{total_runs} ===")
                        print(f"B={B}, N={N}, d={d}, Model={model_label}, Data={data_label}")
                        
                        # Build eigenvalues list for data generation:
                        eigenvals = data_func(d)  # either None (isotropic) or list (non-isotropic)
                        
                        # Run experiments for this configuration (each configuration runs num_experiments seeds)
                        all_logs, agg_data = run_experiments(
                            num_experiments=num_experiments,
                            B=B,
                            N=N,
                            d=d,
                            num_steps=num_steps,
                            metric_interval=metric_interval,
                            val_batch_size=val_batch_size,
                            device=device,
                            compare_mode=compare_mode,
                            activation_choice=activation_choice,
                            leaky_alpha=leaky_alpha,
                            eigenvalues_list=eigenvals,
                            exp_folder="results",
                            noise_variance=noise_variance,
                            line_search_min=line_search_min,
                            line_search_max=line_search_max,
                            line_search_steps=line_search_steps
                        )
    
    print(f"\nTotal configurations launched: {total_runs}")
    print("All experiments completed. Check the 'results/' folder for logs and plots.")