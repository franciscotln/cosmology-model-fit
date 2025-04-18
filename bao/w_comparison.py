import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


def non_linear_equation_of_state(z, params):
    _, w0, wa = params
    return wa + (w0 - wa) * np.exp(-0.5*(1 + z)**2 + 0.5)


def non_linear_normalized_energy_density(zs, params):
    def integrand(z):
        w = non_linear_equation_of_state(z, params)
        return 3 * (1 + w) / (1 + z)

    return np.exp([quad(integrand, 0, z)[0] for z in zs])


def non_linear_hubble(z, params):
    O_m = params[0]
    return np.sqrt(
        O_m * (1 + z)**3 + (1 - O_m)*non_linear_normalized_energy_density(z, params)
    )


def cpl_equation_of_state(z, params):
    _, w0, wa = params
    return w0 + wa * z / (1 + z)


def cpl_normalized_energy_density(z, params):
    _, w0, wa = params
    return (1 + z)**(3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))


def cpl_hubble(z, params):
    O_m = params[0]
    return np.sqrt(
        O_m * (1 + z)**3 + (1 - O_m)*cpl_normalized_energy_density(z, params)
    )


def equation_of_state(z, params, non_linear):
    if non_linear:
        return non_linear_equation_of_state(z, params)
    else:
        return cpl_equation_of_state(z, params)


def normalized_energy_density(z, params, non_linear):
    if non_linear:
        return non_linear_normalized_energy_density(z, params)
    else:
        return cpl_normalized_energy_density(z, params)


def hubble(z, params, non_linear):
    if non_linear:
        return non_linear_hubble(z, params)
    else:
        return cpl_hubble(z, params)


def main(dataset):
    z_min = 0
    z_max = 3.0
    z_range = np.linspace(z_min, z_max, 1000)
    fig, ax = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
    ax[0].set_title(f"{dataset['name']}: non-linear vs CPL w0waCDM")
    ax[0].plot(
        z_range,
        equation_of_state(z_range, dataset["non_linear"]["central"], True),
        label="Non-linear",
        color='red',
    )
    ax[0].fill_between(
        z_range,
        equation_of_state(z_range, dataset["non_linear"]["lower"], True),
        equation_of_state(z_range, dataset["non_linear"]["upper"], True),
        color='red',
        alpha=0.1,
    )
    ax[0].plot(
        z_range,
        equation_of_state(z_range, dataset["CPL"]["central"], False),
        label="CPL",
        color='blue',
    )
    ax[0].fill_between(
        z_range,
        equation_of_state(z_range, dataset["CPL"]["lower"], False),
        equation_of_state(z_range, dataset["CPL"]["upper"], False),
        color='blue',
        alpha=0.1,
    )
    ax[0].set_ylabel(r"$\omega(z)$")
    ax[0].set_ylim(-2.5, 0.0)
    ax[0].axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
    ax[0].legend()

    ax[1].plot(
        z_range,
        normalized_energy_density(z_range, dataset["non_linear"]["central"], True),
        label="Non-linear",
        color='red',
    )
    ax[1].fill_between(
        z_range,
        normalized_energy_density(z_range, dataset["non_linear"]["lower"], True),
        normalized_energy_density(z_range, dataset["non_linear"]["upper"], True),
        color='red',
        alpha=0.1,
    )
    ax[1].plot(
        z_range,
        normalized_energy_density(z_range, dataset["CPL"]["central"], False),
        label="CPL",
        color='blue',
    )
    ax[1].fill_between(
        z_range,
        normalized_energy_density(z_range, dataset["CPL"]["lower"], False),
        normalized_energy_density(z_range, dataset["CPL"]["upper"], False),
        color='blue',
        alpha=0.1,
    )
    ax[1].set_ylabel(r"$\rho(z)$")
    ax[1].set_ylim(0, 2.0)
    ax[1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax[1].legend()

    ax[2].plot(
        z_range,
        normalized_energy_density(z_range, dataset["non_linear"]["central"], True) * equation_of_state(z_range, dataset["non_linear"]["central"], True),
        label="Non-linear",
        color='red',
    )
    ax[2].fill_between(
        z_range,
        normalized_energy_density(z_range, dataset["non_linear"]["lower"], True) * equation_of_state(z_range, dataset["non_linear"]["lower"], True),
        normalized_energy_density(z_range, dataset["non_linear"]["upper"], True) * equation_of_state(z_range, dataset["non_linear"]["upper"], True),
        color='red',
        alpha=0.1,
    )
    ax[2].plot(
        z_range,
        normalized_energy_density(z_range, dataset["CPL"]["central"], False) * equation_of_state(z_range, dataset["CPL"]["central"], False),
        label="CPL",
        color='blue',
    )
    ax[2].fill_between(
        z_range,
        normalized_energy_density(z_range, dataset["CPL"]["lower"], False) * equation_of_state(z_range, dataset["CPL"]["lower"], False),
        normalized_energy_density(z_range, dataset["CPL"]["upper"], False) * equation_of_state(z_range, dataset["CPL"]["upper"], False),
        color='blue',
        alpha=0.1,
    )
    ax[2].set_ylabel("pressure")
    ax[2].set_ylim(-3.0, 0)
    ax[2].axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
    ax[2].legend()

    ax[3].plot(
        z_range,
        hubble(z_range, dataset["non_linear"]["central"], True),
        label="Non-linear",
        color='red',
    )
    ax[3].plot(
        z_range,
        hubble(z_range, dataset["CPL"]["central"], False),
        label="CPL",
        color='blue',
        linestyle='--',
    )
    ax[3].set_xlabel("redshift")
    ax[3].set_ylabel("H(z)")
    ax[3].set_xlim(z_min, z_max)
    ax[3].set_ylim(1.0, 5.0)
    ax[3].legend()
    plt.show()

#  Datasets and fits
desi_union3 = {
    "name": "DESI+Union3",
    "non_linear": {
        "central": [0.3283, -0.7459, -1.2466],
        "lower": [0.3283 - 0.0169, -0.7459 + 0.0907, -1.2466 + 0.2124],
        "upper": [0.3283 + 0.0152, -0.7459 - 0.0848, -1.2466 - 0.2139],
    },
    "CPL": {
        "central": [0.3304, -0.7002, -0.9954],
        "lower": [0.3304 - 0.0163, -0.7002 + 0.1128, -0.9954 + 0.5110],
        "upper": [0.3304 + 0.0155, -0.7002 - 0.1004, -0.9954 - 0.5520],
    }
}

desi_des5y = {
    "name": "DESI+DES5Y",
    "non_linear": {
        "central": [0.3179, -0.8230, -1.1564],
        "lower": [0.3179 - 0.0159, -0.8230 + 0.0571, -1.1564 + 0.2069],
        "upper": [0.3179 + 0.0136, -0.8230 - 0.0528, -1.1564 - 0.2082],
    },
    "CPL": {
        "central": [0.3210, -0.7878, -0.7034],
        "lower": [0.3210 - 0.0132, -0.7878 + 0.0697, -0.7034 + 0.3901],
        "upper": [0.3210 + 0.0122, -0.7878 - 0.0607, -0.7034 - 0.4490],
    }
}

desi_pantheon = {
    "name": "DESI+Pantheon+",
    "non_linear": {
        "central": [0.3047, -0.8960, -1.0303],
        "lower": [0.3047 - 0.0182, -0.8960 + 0.0498, -1.0177 + 0.2076],
        "upper": [0.3047 + 0.0142, -0.8960 - 0.0483, -1.0177 - 0.1969],
    },
    "CPL": {
        "central": [0.3021, -0.8964,  -0.1376],
        "lower": [0.3021 - 0.0228, -0.8964 + 0.0607,  -0.1376 + 0.4676],
        "upper": [0.3021 + 0.0154, -0.8964 - 0.0561,  -0.1376 - 0.4503],
    }
}

if __name__ == "__main__":
    main(desi_des5y)
