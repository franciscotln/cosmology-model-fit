import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.constraints import Interval
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
        self.covar_module = ScaleKernel(
            RBFKernel(lengthscale_constraint=Interval(1.2 * z.max(), 3 * z.max()))
        )
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

training_iterations = 5000
for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(z_values)
    loss = -mll(output, H_values)
    loss.backward()
    (
        print(
            "Iter %d/%d - Loss: %.4f | output scale: %.4f | length scale: %.4f | noise scale: %.3f"
            % (
                i + 1,
                training_iterations,
                loss.item() - H.size * np.log(h_std),
                model.covar_module.outputscale.item() * h_std**2,
                model.covar_module.base_kernel.lengthscale.item(),  # 2.366
                model.likelihood.noise_covar.noise_scale.item(),  # 0.699^2
            )
        )
        if i % 10 == 0
        else None
    )
    optimizer.step()

model.eval()
likelihood.eval()

X_test = torch.linspace(0, z.max(), 100)
test_noise = torch.full_like(X_test, 0.0001)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(X_test), noise=test_noise)
    H_pred = observed_pred.mean
    pred_var = observed_pred.variance
    cov_pred = observed_pred.covariance_matrix * h_std**2
    pred_std = torch.sqrt(pred_var) * h_std
    H_pred = H_pred * h_std + h_mean
    scale = model.likelihood.noise_covar.noise_scale.item() ** 2


X_test.requires_grad_(True)
H_mean_autograd = model(X_test).mean * h_std + h_mean
dH_dz = torch.autograd.grad(H_mean_autograd.sum(), X_test, create_graph=True)[0]

q_z = -1 + (1 + X_test) * (dH_dz / H_mean_autograd)
X_test = X_test.detach().numpy()
q_z = q_z.detach().numpy()

plt.style.use("mpl20")
plt.plot(X_test, q_z, label="GP q(z)")
plt.axhline(0, color="gray", linestyle="--")
plt.xlabel("z")
plt.ylabel("q(z)")
plt.legend()
plt.title("Deceleration Parameter from GP")
plt.grid(True)
plt.savefig("cosmic_chronometers/cc_gp_qz.png", dpi=300)
plt.close()


plt.errorbar(
    x=z, y=H, yerr=np.sqrt(np.diag(cov_mat / scale)), fmt=".", label="CCH", capsize=2
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
plt.savefig("cosmic_chronometers/cc_gp_Hz.png", dpi=300)
plt.close()

plt.imshow(cov_pred, cmap="hot", interpolation="none")
plt.colorbar()
plt.title("Covariance Matrix")
plt.show()
