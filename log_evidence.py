import numpy as np
import numdifftools as nd
import scipy.optimize


# Laplace approximation for Bayesian evidence (ln Z) using Hessian
def log_evidence(mc_samples, log_probs, log_probability, bounds):
    """
    Laplace approximation for Bayesian evidence (ln Z) using Hessian at MAP.
    - -inf < ln(Z) < 1: weak
    - 1 <= ln(Z) < 3: positive
    - 3 <= ln(Z) < 5: strong
    - ln(Z) >= 5: very strong
    """
    # Find best MCMC sample as starting point
    best_sample_idx = np.argmax(log_probs)
    initial_guess = mc_samples[best_sample_idx]
    n_params = initial_guess.shape[0]

    def objective_function(theta):
        """Negative log probability for minimization, handling infinities."""
        lp = log_probability(theta)
        if np.isinf(lp) or np.isnan(lp):
            return 1e10
        return -lp

    best_result = scipy.optimize.minimize(
        objective_function,
        x0=initial_guess,
        bounds=bounds,
    )
    best_log_prob = -best_result.fun
    initial_log_prob = log_probs[best_sample_idx]

    # If optimization didn't converge, just use the best MCMC sample
    if not best_result.success:
        print("Optimization did not converge, using best MCMC sample for Hessian.")
        theta_map = initial_guess
        log_post_map = initial_log_prob
    else:
        theta_map = best_result.x
        log_post_map = best_log_prob

    # Wrap log_probability to handle infinities (for numerical derivatives)
    def log_prob_for_hessian(theta):
        lp = log_probability(theta)
        # Replace -inf with large negative number for numerical stability
        if np.isinf(lp):
            return -1e10
        return lp

    try:
        hessian_calculator = nd.Hessian(log_prob_for_hessian, step=1e-5)
        H = hessian_calculator(theta_map)
        neg_H = -H

        # Check and fix positive definiteness
        eigenvalues = np.linalg.eigvalsh(neg_H)
        min_eig = np.min(eigenvalues)

        if min_eig <= 0:
            jitter = abs(min_eig) + 1e-6 * np.max(np.abs(eigenvalues))
            neg_H += jitter * np.eye(n_params)

        sign, logdet = np.linalg.slogdet(neg_H)

        if sign <= 0:
            print("Warning: Hessian is still not positive definite after jitter!")
            return np.nan

        # Laplace approximation ln(Z)
        ln_z = log_post_map + 0.5 * n_params * np.log(2 * np.pi) - 0.5 * logdet

        return ln_z

    except Exception as e:
        print(f"Error computing Hessian: {e}")
        return np.nan


"""
Scales describe in:
https://ptfonseca.github.io/pcal/reference/bfactor_log_interpret.html

Kass and Raftery (1995) suggest the following scale for interpreting the
difference in log-evidence (ln Z) between two models:
-inf < ln(Z) < 1: weak
1 <= ln(Z) < 3: positive
3 <= ln(Z) < 5: strong
ln(Z) >= 5: very strong

Jeffreys (1961) suggested the interpretation of Bayes factors
in half-units on the base 10 logarithmic scale, as indicated in the following:
0 < log10(z) < 0.5: weak
0.5 < log10(z) < 1: substantial
1 < log10(z) < 1.5: strong
1.5 < log10(z) < 2.0: very strong
log10(z) > 2.0: decisive
"""
