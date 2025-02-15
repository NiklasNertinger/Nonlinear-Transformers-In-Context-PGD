"""
This script processes experimental results by:
1. Crawling directories to collect "aggregated_best_metrics.csv" files.
2. Extracting relevant experimental parameters from directory names.
3. Creating a summary table of in-context loss values and PGD baseline losses.
4. Formatting and structuring results for LaTeX export.

Output:
- A LaTeX-formatted table summarizing in-context loss results across different setups.
"""

#############################################
# 1. SETUP & HELPER FUNCTIONS
#############################################

import os
import re
import pandas as pd

def crawl_best_metrics(base_dir):
    """
    Recursively searches for files named 'aggregated_best_metrics.csv' in base_dir.
    For each such file, it:
      - Extracts experiment parameters (activation "act", noise, cond) from the parent folder name.
      - Loads the CSV file (which must contain at least 'in_context_loss' and 'pgd_loss')
        and extracts the first row’s values.
    Returns a DataFrame with columns:
      act, noise, cond, in_context_loss, pgd_loss, file_path
    """
    data = []
    # Regex pattern for a folder name like:
    # exp_N20_d5_act_leakyrelu0.25_noise0.00_cond100.00_diagCov_False_steps10000_valB10000_B32_int50_ls(0.01-2-200)_cmp_manual
    # We ignore diagCov here.
    pattern = re.compile(
        r"exp_N\d+_d\d+_act_(?P<act>[^_]+)_noise(?P<noise>[\d\.]+)_cond(?P<cond>[\d\.]+)_diagCov_[^_]+_steps\d+_valB\d+_B\d+_int\d+_ls\([\d\.-]+\)_cmp_\w+"
    )
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "aggregated_best_metrics.csv":
                full_path = os.path.join(root, file)
                # Assume the CSV file is inside an "aggregated" folder; its parent folder holds the parameters.
                exp_folder = os.path.basename(os.path.dirname(os.path.dirname(full_path)))
                match = pattern.search(exp_folder)
                if not match:
                    print(f"Warning: folder {exp_folder} did not match expected pattern.")
                    continue
                params = match.groupdict()
                try:
                    df_csv = pd.read_csv(full_path)
                except Exception as e:
                    print(f"Error reading {full_path}: {e}")
                    continue
                if 'in_context_loss' not in df_csv.columns or 'pgd_loss' not in df_csv.columns:
                    print(f"File {full_path} missing required columns.")
                    continue
                # Extract the first row’s values.
                params['in_context_loss'] = df_csv['in_context_loss'].iloc[0]
                params['pgd_loss'] = df_csv['pgd_loss'].iloc[0]
                params['file_path'] = full_path
                data.append(params)
    return pd.DataFrame(data)

def create_summary_table(df):
    """
    Given a DataFrame from crawl_best_metrics, creates a pivot table where:
      - Rows: activation model (act)
      - Columns: a string combination of noise and cond (formatted as "noise=<noise>, cond=<cond>")
      - Values: in_context_loss
    Also creates an extra row "pgd" which, for each column, contains the best (lowest) pgd_loss among all activations.
    Duplicate entries are aggregated using the minimum.
    """
    # Create a new column representing the combination (only noise and cond).
    df['combination'] = df.apply(lambda r: f"noise={r['noise']}, cond={r['cond']}", axis=1)
    
    # Use pivot_table with an aggregation function (min) to resolve duplicates.
    pivot_in = df.pivot_table(index='act', columns='combination', values='in_context_loss', aggfunc='min')
    pivot_pgd = df.pivot_table(index='act', columns='combination', values='pgd_loss', aggfunc='min')
    
    # For each combination, choose the minimum pgd_loss as the baseline.
    pgd_baseline = pivot_pgd.min(axis=0)
    df_pgd = pd.DataFrame(pgd_baseline).T
    df_pgd.index = ['pgd']
    
    summary = pd.concat([pivot_in, df_pgd])
    return summary

def create_multilevel_columns(col_str):
    """
    Given a column string of the form:
      "noise=<noise>, cond=<cond>"
    returns a tuple:
      (Condition number, Noise)
    where:
      - Condition number is formatted as "$\\kappa=<X>$" (with X as an integer)
      - Noise is just the noise number formatted with three decimals (as a string)
    """
    pattern = r"noise=([\d\.]+),\s*cond=([\d\.]+)"
    m = re.search(pattern, col_str)
    if not m:
        return ("", "")
    noise_val, cond_val = m.groups()
    cond_int = int(float(cond_val))
    condition = f"$\\kappa={cond_int}$"
    noise_float = float(noise_val)
    noise_str = f"{noise_float:.3f}"
    return (condition, noise_str)

def add_multilevel_columns(summary_table):
    """
    Converts the summary table's single-level column names into a MultiIndex.
    """
    new_tuples = [create_multilevel_columns(col) for col in summary_table.columns]
    # Create a MultiIndex with two levels: Condition number and Noise.
    multi = pd.MultiIndex.from_tuples(new_tuples, names=["Condition number", "Noise $\\sigma^2$"])
    summary_table.columns = multi
    return summary_table

def sort_columns(multicol_df):
    """
    Sorts the MultiIndex columns by condition number and noise variance.
    """
    def sort_key(col_tuple):
        condition, noise = col_tuple
        cond_num = int(re.search(r"\\kappa=(\d+)", condition).group(1))
        noise_val = float(noise)
        return (cond_num, noise_val)
    sorted_cols = sorted(multicol_df.columns, key=sort_key)
    return multicol_df.reindex(columns=sorted_cols)

def format_and_bold_min(summary_table):
    """
    Returns a DataFrame where each numeric cell is formatted to two decimals.
    In each column, the minimum value is wrapped in \textbf{...} so that it appears bold in LaTeX.
    """
    formatted = summary_table.copy()
    for col in summary_table.columns:
        col_min = summary_table[col].min()
        def format_cell(x):
            if pd.isna(x):
                return ""
            if abs(x - col_min) < 1e-6:
                return r"\textbf{" + f"{x:.2f}" + "}"
            else:
                return f"{x:.2f}"
        formatted[col] = summary_table[col].apply(format_cell)
    return formatted

#############################################
# 2. MAIN SCRIPT: CRAWL, PROCESS & EXPORT
#############################################

if __name__ == "__main__":
    # Set the base directory where your experiment results are stored.
    base_dir = r"..\v3_results"  # adjust as needed
    
    # Crawl the directory structure and collect the data.
    df_crawled = crawl_best_metrics(base_dir)
    print("Crawled Data:")
    print(df_crawled)
    
    # Create the summary table.
    summary = create_summary_table(df_crawled)
    
    # Change row names to proper case.
    row_mapping = {
        "leakyrelu0.25": "LeakyReLU0.25",
        "leakyrelu0.5": "LeakyReLU0.5",
        "leakyrelu0.75": "LeakyReLU0.75",
        "relu": "ReLU",
        "linear": "Linear",
        "softmax": "Softmax",
        "pgd": "PGD"
    }
    summary.index = summary.index.map(lambda x: row_mapping.get(x.lower(), x))
    
    # Reorder rows in the desired order.
    desired_order = ["LeakyReLU0.25", "LeakyReLU0.5", "LeakyReLU0.75", "ReLU", "Linear", "Softmax", "PGD"]
    summary = summary.reindex(desired_order)
    
    # Convert the column index to a MultiIndex with our desired labels.
    summary = add_multilevel_columns(summary)
    summary = sort_columns(summary)
    
    # Format the numbers and bold the minimum in each column.
    formatted_summary = format_and_bold_min(summary)
    
    # Export the table to LaTeX.
    latex_code = formatted_summary.to_latex(multicolumn=True,
                                            multicolumn_format='c',
                                            escape=False,
                                            caption="Summary Metrics Table",
                                            label="tab:summary")
    with open("summary_table.tex", "w") as f:
        f.write(latex_code)
    
    print("LaTeX table code saved to summary_table.tex")
