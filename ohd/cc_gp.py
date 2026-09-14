import numpy as np
import matplotlib.pyplot as plt
import torch
from gpytorch.models import ExactGP
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.settings import fast_pred_var, detach_test_caches
from .gp_lib import FixedNoiseGaussianLikelihood
from y2005cc.data import get_data

legend, z, H, cov_mat = get_data()

h_mean = np.mean(H)
h_std = np.std(H)

z_values = torch.tensor(z, dtype=torch.float32).reshape(-1)
H_values = torch.tensor((H - h_mean) / h_std, dtype=torch.float32).reshape(-1)
cov_matrix = torch.tensor(cov_mat / h_std**2, dtype=torch.float32)


class HubbleGaussianProcess(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(HubbleGaussianProcess, self).__init__(train_x, train_y, likelihood)
        self.covar_module = ScaleKernel(RBFKernel())
        self.mean_module = ConstantMean()

    def forward(self, x):
        return MultivariateNormal(
            mean=self.mean_module(x), covariance_matrix=self.covar_module(x)
        )


likelihood = FixedNoiseGaussianLikelihood(
    noise=cov_matrix, learn_additional_noise=False, learn_noise_scale=True
)
model = HubbleGaussianProcess(train_x=z_values, train_y=H_values, likelihood=likelihood)

model.train()
likelihood.train()

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
mll = ExactMarginalLogLikelihood(model.likelihood, model)


# --- Hyperparameters for Dynamic Schedule ---
# Ensure the deceleration parameter q(z) remains monotonic during training
lambda_mono = 0.01         # Start small so GP learns overall H(z) scale first
lambda_max = 1000.0        # Upper bound to prevent numerical explosion
growth_factor = 1.05       # Rate of penalty increase (5% per update)
decay_factor = 0.98        # Optional rate to decay penalty if constraint is satisfied
check_interval = 50        # How often (in iterations) to evaluate and update lambda
tolerance = 5e-4           # Acceptable numerical threshold for zero violation

training_iterations = 15_000
z_grid = torch.linspace(0.0, float(z.max()), 1000, dtype=torch.float32)

for i in range(training_iterations):
    model.train()
    likelihood.train()
    optimizer.zero_grad()

    # Standard MLL loss
    output = model(z_values)
    mll_loss = -mll(output, H_values)

    # Predictive Grid Penalty step
    model.eval()
    likelihood.eval()
    z_g = z_grid.clone().detach().requires_grad_(True)

    with fast_pred_var(False), detach_test_caches(False):
        H_pred_grid = model(z_g).mean * h_std + h_mean

    dH_dz = torch.autograd.grad(outputs=H_pred_grid.sum(), inputs=z_g, create_graph=True)[0]
    q_grid = -1.0 + (1.0 + z_g) * (dH_dz / H_pred_grid)

    q_steps = q_grid[1:] - q_grid[:-1]
    penalty = torch.relu(-q_steps).sum()

    # Compute joint loss using CURRENT lambda_mono
    total_loss = mll_loss + lambda_mono * penalty
    
    # Backpropagate & step optimizer
    total_loss.backward()
    optimizer.step()

    # --- DYNAMIC LAMBDA UPDATE STEP ---
    if i > 0 and i % check_interval == 0:
        penalty_val = penalty.item()
        
        if penalty_val > tolerance:
            # Violation exists -> increase pressure
            lambda_mono = min(lambda_mono * growth_factor, lambda_max)
        else:
            # Constraint satisfied -> slightly relax to prioritize likelihood fit
            lambda_mono = max(lambda_mono * decay_factor, 1e-4)

    # Iter 14000/15000 | MLL: 0.6114 | Penalty: 0.000349 | lambda_mono: 3.0707
    if i % 1000 == 0:
        print(
            f"Iter {i}/{training_iterations} | "
            f"MLL: {mll_loss.item():.4f} | "
            f"Penalty: {penalty.item():.6f} | "
            f"lambda_mono: {lambda_mono:.4f}"
        )

model.eval()
likelihood.eval()

X_test = torch.linspace(0, z.max(), 100)
test_noise = torch.full_like(X_test, 0.0001)

with torch.no_grad(), fast_pred_var():
    observed_pred = likelihood(model(X_test), noise=test_noise)
    H_pred = observed_pred.mean
    pred_var = observed_pred.variance
    cov_pred = observed_pred.covariance_matrix * h_std**2
    pred_std = torch.sqrt(pred_var) * h_std
    H_pred = H_pred * h_std + h_mean
    scale = model.likelihood.noise_covar.noise_scale.item()


X_test.requires_grad_(True)
H_mean_autograd = model(X_test).mean * h_std + h_mean
dH_dz = torch.autograd.grad(H_mean_autograd.sum(), X_test, create_graph=True)[0]

q_z = -1 + (1 + X_test) * (dH_dz / H_mean_autograd)
X_test = X_test.detach().numpy()
q_z = q_z.detach().numpy()

plt.style.use("mpl20")
plt.plot(X_test, q_z, label="GP q(z)")
plt.axhline(0, color="gray", linestyle="--")
plt.axhline(0.5, color="gray", linestyle="--")
plt.xlabel("z")
plt.ylabel("q(z)")
plt.legend()
plt.title("Deceleration Parameter from GP")
plt.grid(True)
plt.savefig("ohd/cc_gp_qz.png", dpi=300)
plt.close()


plt.errorbar(
    x=z,
    y=H,
    yerr=scale * np.sqrt(np.diag(cov_mat)),
    fmt=".",
    label="CCH",
    capsize=2,
)
plt.plot(X_test, H_pred, label="GP Mean")
plt.fill_between(
    X_test, H_pred - pred_std, H_pred + pred_std, alpha=0.5, label=r"$1\sigma$"
)
plt.fill_between(
    X_test,
    H_pred - 2 * pred_std,
    H_pred + 2 * pred_std,
    alpha=0.3,
    label=r"$2\sigma$",
)
plt.xlim(0, 2)
plt.xlabel("z")
plt.ylabel("H(z) [km/s/Mpc]")
plt.title(f"GP $H_0$: {H_pred[0]:.1f} ± {pred_std[0]:.1f} km/s/Mpc")
plt.legend()
plt.grid(True)
plt.savefig("ohd/cc_gp_Hz.png", dpi=300)
plt.close()

plt.imshow(cov_pred, cmap="hot", interpolation="none")
plt.colorbar()
plt.title("Covariance Matrix")
plt.show()
