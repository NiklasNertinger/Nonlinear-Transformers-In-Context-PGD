"""
This script processes training logs saved as .pt files, converting them to .csv format 
and generating uncertainty plots for various metrics. 

### Functionality:
1. Converts .pt files in a specified folder to .csv.
2. Extracts relevant training metrics and creates summary plots.
3. Saves plots in both PNG and TikZ formats.

### Assumptions:
- The .pt files should be named according to one of the following model types:
  "linear", "relu", "leakyrelu0.25", "leakyrelu0.5", "leakyrelu0.75", "softmax".
- The files are placed in a folder together manually by the user beforehand.
- The following metrics (with standard deviations) are available in the .csv:
  - `in_context_loss_mean.mean`, `in_context_loss_mean.std`
  - `pgd_loss_mean.mean`, `pgd_loss_mean.std`
  - `l2_pred_diff_mean.mean`, `l2_pred_diff_mean.std`
  - `cosine_sens_mean.mean`, `cosine_sens_mean.std`
  - `l2_sens_diff_mean.mean`, `l2_sens_diff_mean.std`
"""

#############################################
# 1. SETUP AND HELPER FUNCTIONS
#############################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tikzplotlib
import webcolors
import torch
import csv

# Patch webcolors to provide CSS3_HEX_TO_NAMES if missing.
if not hasattr(webcolors, 'CSS3_HEX_TO_NAMES'):
    try:
        webcolors.CSS3_HEX_TO_NAMES = webcolors.HTML5_HEX_TO_NAMES
        print("Patched webcolors: CSS3_HEX_TO_NAMES set to HTML5_HEX_TO_NAMES.")
    except AttributeError:
        webcolors.CSS3_HEX_TO_NAMES = {}
        print("Patched webcolors: CSS3_HEX_TO_NAMES not available; set to an empty dictionary.")

def flatten_dict(d, parent_key='', sep='.'):
    """
    Recursively flattens a nested dictionary.
    
    For example:
      {"a": 1, "b": {"c": 2, "d": 3}}  => {"a": 1, "b.c": 2, "b.d": 3}
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def convert_value_to_array(value):
    """
    Converts a value to a 1D NumPy array.
    
    - If it's a torch.Tensor, converts to numpy.
    - If it's a list or tuple, converts to numpy.
    - If it's a scalar, returns a one-element array.
    - Otherwise, attempts np.array conversion.
    """
    if isinstance(value, torch.Tensor):
        arr = value.cpu().numpy()
    elif isinstance(value, (list, tuple)):
        arr = np.array(value)
    else:
        arr = np.array(value)
    
    # Ensure the array is 1D. If it has more dimensions, flatten it.
    if arr.ndim > 1:
        arr = arr.flatten()
    elif arr.ndim == 0:
        arr = np.array([arr])
    return arr

def write_csv(csv_filepath, headers, rows):
    """
    Writes headers and rows to a CSV file with error handling.
    """
    try:
        with open(csv_filepath, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"Converted to CSV: '{csv_filepath}'")
    except Exception as e:
        print(f"Error writing CSV for {csv_filepath}: {e}")

def convert_pt_to_csv(pt_filepath, csv_filepath):
    """
    Loads a .pt file and writes its (flattened) contents to a CSV file.
    
    The file is assumed to contain a dictionary (possibly nested) with
    keys that map to numeric data (scalars or lists/tensors).
    """
    try:
        data = torch.load(pt_filepath, map_location="cpu")
    except Exception as e:
        print(f"Failed to load {pt_filepath}: {e}")
        return

    # If the data is a tensor, simply convert it to CSV.
    if isinstance(data, torch.Tensor):
        arr = data.cpu().numpy()
        np.savetxt(csv_filepath, arr, delimiter=",", fmt='%s')
        print(f"Converted tensor from {pt_filepath} to {csv_filepath}")
        return

    # If the data is a dictionary, flatten it.
    if isinstance(data, dict):
        flat_data = flatten_dict(data)
    else:
        print(f"Unsupported data type in {pt_filepath}: {type(data)}")
        return

    # For each key, try to convert its value to a 1D numpy array.
    converted = {}
    for key, value in flat_data.items():
        try:
            arr = convert_value_to_array(value)
            converted[key] = arr
        except Exception as e:
            print(f"Skipping key '{key}' in {pt_filepath} (could not convert value: {e})")
    
    if not converted:
        print(f"No convertible columns found in {pt_filepath}")
        return

    # Determine the length to use for CSV rows.
    lengths = [arr.shape[0] for arr in converted.values()]
    max_length = max(lengths)

    # Check for consistency: if an array is not length 1 and not equal to max_length,
    # then we cannot easily combine the data.
    for key, arr in converted.items():
        if arr.shape[0] not in (1, max_length):
            print(f"Skipping {pt_filepath}: Key '{key}' has length {arr.shape[0]} which is inconsistent with max length {max_length}.")
            return

    # Replicate arrays that are scalars (length 1) to match max_length.
    for key, arr in converted.items():
        if arr.shape[0] == 1 and max_length > 1:
            converted[key] = np.repeat(arr, max_length)

    # Prepare CSV rows: header followed by each row of data.
    headers = list(converted.keys())
    rows = []
    for i in range(max_length):
        row = [converted[key][i] for key in headers]
        rows.append(row)

    write_csv(csv_filepath, headers, rows)

def convert_all_pt_to_csv(folder):
    """
    Iterates over all .pt files in the specified folder and converts each to a CSV.
    The CSV file will have the same base filename and will be placed in the same folder.
    """
    if not os.path.isdir(folder):
        print(f"Folder '{folder}' does not exist.")
        return

    for filename in os.listdir(folder):
        if filename.endswith('.pt'):
            pt_filepath = os.path.join(folder, filename)
            csv_filename = os.path.splitext(filename)[0] + '.csv'
            csv_filepath = os.path.join(folder, csv_filename)
            print(f"Converting {pt_filepath} to {csv_filepath} ...")
            convert_pt_to_csv(pt_filepath, csv_filepath)

# Second webcolors patch (as in the original)
if not hasattr(webcolors, 'CSS3_HEX_TO_NAMES'):
    try:
        webcolors.CSS3_HEX_TO_NAMES = webcolors.HTML5_HEX_TO_NAMES
        print("Patched webcolors: CSS3_HEX_TO_NAMES set to HTML5_HEX_TO_NAMES.")
    except AttributeError:
        webcolors.CSS3_HEX_TO_NAMES = {}
        print("Patched webcolors: CSS3_HEX_TO_NAMES not available; set to an empty dictionary.")

#############################################
# 2. CORE FUNCTIONS (PLOTTING & EXPORT)
#############################################

def save_plot_as_png_and_tikz(folder, base_filename):
    """
    Saves the current matplotlib figure as a PNG and a TikZ file.
    Includes error handling for the TikZ export.
    """
    png_path = os.path.join(folder, f"{base_filename}.png")
    tikz_path = os.path.join(folder, f"{base_filename}.tikz")
    plt.savefig(png_path)
    print(f"Plot saved as PNG in '{png_path}'.")

    try:
        tikzplotlib.save(tikz_path, extra_axis_parameters={'legend style': '{legend columns=1}'})
        print(f"Plot saved as TikZ in '{tikz_path}'.")
    except Exception as e:
        print(f"Error saving TikZ file with legend: {e}")
        ax = plt.gca()
        legend_handles, legend_labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        try:
            tikzplotlib.save(tikz_path, extra_axis_parameters={'legend style': '{legend columns=1}'})
            print(f"Plot saved as TikZ (without legend) in '{tikz_path}'.")
        except Exception as e2:
            print(f"Error saving TikZ file even after removing legend: {e2}")
        if legend_handles and legend_labels:
            ax.legend(legend_handles, legend_labels, fontsize='small', ncol=1)

def create_uncertainty_plot_from_csvs(
    folder, 
    step_multiplier=25, 
    cutoff=2000, 
    include_uncertainty=True, 
    y_min=None,
    y_max=None,
    uncertainty_factor=1.0,
    metric_mean="cosine_sens_mean.mean",
    metric_std="cosine_sens_mean.std",
    metric_label="",     # y-axis description (if empty, no label)
    plot_title="",       # plot headline (if empty, no title)
    x_label="",          # x-axis description (if empty, no label)
    include_legend=True
):
    """
    Creates a plot from CSV files in the given folder and saves both a PNG and a TikZ file.
    Optional parameters let you set the plot title, axis labels, and y-axis limits.
    
    The output files are named after the base name of the folder.
    """
    model_name_mapping = {
        "linear": "Linear",
        "leakyrelu0.75": "LeakyReLU0.75",
        "leakyrelu0.5": "LeakyReLU0.5",
        "leakyrelu0.25": "LeakyReLU0.25",
        "relu": "ReLU",
        "softmax": "Softmax"
    }
    
    csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in folder '{folder}'.")
        return

    plt.figure(figsize=(10, 6))
    for csv_file in csv_files:
        file_path = os.path.join(folder, csv_file)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

        if "step" in df.columns:
            x = df["step"].to_numpy()
        else:
            x = np.arange(len(df))
        x = x * step_multiplier

        if metric_mean not in df.columns or metric_std not in df.columns:
            print(f"Skipping {csv_file}: required columns '{metric_mean}' and/or '{metric_std}' missing.")
            continue

        y = df[metric_mean].to_numpy()
        y_std = df[metric_std].to_numpy()
        mask = x <= cutoff
        x = x[mask]
        y = y[mask]
        y_std = y_std[mask]
        if len(x) == 0:
            print(f"All data in {csv_file} is beyond the cutoff of {cutoff}. Skipping.")
            continue

        raw_label = os.path.splitext(csv_file)[0].lower()
        label = model_name_mapping.get(raw_label, raw_label)
        plt.plot(x, y, label=label)
        if include_uncertainty:
            plt.fill_between(x, y - uncertainty_factor * y_std, y + uncertainty_factor * y_std, alpha=0.3)

    if include_legend:
        plt.legend(fontsize='small', ncol=1)
    if x_label:
        plt.xlabel(x_label, labelpad=5)
    if metric_label:
        plt.ylabel(metric_label, labelpad=5)
    if plot_title:
        plt.title(plot_title)
    plt.tick_params(axis='both', which='major', pad=3)
    plt.tight_layout()
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    elif y_min is not None:
        current_ylim = plt.ylim()
        plt.ylim(y_min, current_ylim[1])
    elif y_max is not None:
        current_ylim = plt.ylim()
        plt.ylim(current_ylim[0], y_max)

    folder_base = os.path.basename(os.path.normpath(folder))
    save_plot_as_png_and_tikz(folder, folder_base)
    plt.show()

#############################################
# 3. MAIN SCRIPT EXECUTION
#############################################

if __name__ == "__main__":
    # Example call using all parameters for the "cosine_sim_isotropic" folder.
    folder = r"..\cosine_sim_isotropic"
    convert_all_pt_to_csv(folder)
    create_uncertainty_plot_from_csvs(
        folder, 
        step_multiplier=50, 
        cutoff=5000, 
        include_uncertainty=True,
        uncertainty_factor=1.0,
        include_legend=True,
        metric_label="Sensitivity Cosine Similarity",
        plot_title="Isotropic",
        y_min=0.38,
        y_max=1.02
    )
