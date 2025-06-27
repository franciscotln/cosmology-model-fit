# # https://arxiv.org/abs/2008.08586
# # https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/642/A150#/browse
import pandas as pd
import numpy as np

df = pd.read_csv("y2018quasars/raw-data/data.txt", sep="\s+").sort_values(by="z")

mask = df["z"] >= 0


def get_data():
    return (
        f"Quasars: ({np.sum(mask)} objects)",
        df["z"][mask],
        df["DM"][mask],
        df["e_DM"][mask],
    )


def bin_equally_populated(z, mu, sigma_mu_ind, n_bins=48, sigma_mu_type="sem_errors"):
    """
    Bins data into equally populated bins based on the number of data points.

    Args:
        z (pd.Series): Redshift values, assumed to be already sorted.
        mu (pd.Series): Distance Modulus values.
        sigma_mu_ind (pd.Series): Individual errors on Distance Modulus (e_DM).
        n_bins (int): The desired number of bins.
        sigma_mu_type (str): Specifies how to calculate the binned sigma_mu.
                             Options:
                             - 'sem_errors': Standard Error of the Mean, propagating individual errors.
                                             Formula: sqrt(sum(e_DM^2)) / N_bin
                             - 'sem_data': Standard Error of the Mean, based on the spread of mu values.
                                           Formula: std(mu_values_in_bin) / sqrt(N_bin)
                             - 'rms_errors': Root Mean Square of individual errors.
                                             Formula: sqrt(sum(e_DM^2) / N_bin)

    Returns:
        pd.DataFrame: A DataFrame with binned 'z', 'mu', and 'sigma_mu' values.
    """
    df_bin = (
        pd.DataFrame({"z": z, "mu": mu, "sigma_mu_ind": sigma_mu_ind})
        .sort_values("z")
        .reset_index(drop=True)
    )

    df_bin["bin"] = pd.qcut(df_bin["z"], q=n_bins, labels=False, duplicates="drop")
    actual_n_bins = df_bin["bin"].nunique()

    if actual_n_bins < n_bins:
        print(
            f"Warning: Number of bins reduced from {n_bins} to {actual_n_bins} due to duplicate z values."
        )
        n_bins = actual_n_bins

    def calculate_binned_sigma_mu(errors_or_data, bin_type):
        N_bin = len(errors_or_data)
        if N_bin == 0:
            return np.nan
        if bin_type == "sem_errors":
            return np.sqrt(np.sum(errors_or_data**2)) / N_bin
        elif bin_type == "sem_data":
            return np.std(errors_or_data) / np.sqrt(N_bin)
        elif bin_type == "rms_errors":
            return np.sqrt(np.sum(errors_or_data**2) / N_bin)
        else:
            raise ValueError("Invalid sigma_mu_type provided.")

    return (
        df_bin.groupby("bin")
        .agg(
            {
                "z": "mean",
                "mu": "mean",
                "sigma_mu_ind": lambda x: calculate_binned_sigma_mu(x, sigma_mu_type),
            }
        )
        .rename(columns={"sigma_mu_ind": "sigma_mu"})
        .reset_index(drop=True)
    )


def get_binned_data(n_bins=48, sigma_mu_type="sem_errors"):
    _, z, mu, sigma_mu_ind = get_data()

    binned_data = bin_equally_populated(z, mu, sigma_mu_ind, n_bins, sigma_mu_type)

    legend = f"Quasars - {binned_data.shape[0]} bins"

    return (legend, binned_data["z"], binned_data["mu"], binned_data["sigma_mu"])
