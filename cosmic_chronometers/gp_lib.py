from typing import Any, Optional, Union
from copy import deepcopy
import torch
import gpytorch
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods.noise_models import HomoskedasticNoise
from gpytorch.lazy import DiagLazyTensor, NonLazyTensor, ZeroLazyTensor, LinearOperator
import gpytorch.settings as settings
import gpytorch.utils.warnings as warnings


class FixedGaussianNoise(gpytorch.Module):
    def __init__(self, noise: torch.Tensor, learn_noise_scale: bool = False) -> None:
        """
        noise can be either a 1D tensor (for diagonal noise)
        or a 2D symmetric positive semi-definite matrix (for full cov).
        """
        super().__init__()

        min_noise = settings.min_fixed_noise.value(noise.dtype)
        if noise.ndim == 2:
            if not torch.allclose(noise, noise.transpose(-1, -2)):
                raise ValueError("Covariance matrix must be symmetric.")
            diag = noise.diag()
            if diag.lt(min_noise).any():
                warnings.warn(
                    "Very small noise values detected on the diagonal. Rounding small noise values up.",
                    warnings.NumericalWarning,
                )
                noise = noise.clone()
                noise.diagonal().clamp_min_(min_noise)
        elif noise.ndim == 1:
            if noise.lt(min_noise).any():
                warnings.warn(
                    "Very small noise values detected. Rounding small noise values up.",
                    warnings.NumericalWarning,
                )
                noise = noise.clamp_min(min_noise)
        else:
            raise ValueError("Noise must be either a vector or a square matrix.")

        self.noise = noise

        self.register_parameter(
            "noise_scale",
            torch.nn.Parameter(torch.ones(()), requires_grad=learn_noise_scale),
        )

    def forward(
        self,
        *params: Any,
        shape: Optional[torch.Size] = None,
        noise: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> LinearOperator:
        raw_noise = noise if noise is not None else self.noise
        scaled_noise = raw_noise * self.noise_scale**2

        if scaled_noise.ndim == 2:
            return NonLazyTensor(scaled_noise)
        else:
            return DiagLazyTensor(scaled_noise)

    def _apply(self, fn):
        self.noise = fn(self.noise)
        return super()._apply(fn)


class FixedNoiseGaussianLikelihood(_GaussianLikelihoodBase):
    r"""
    A Likelihood that assumes fixed heteroscedastic noise. This is useful when you have fixed, known observation
    noise for each training example.

    Note that this likelihood takes an additional argument when you call it, `noise`, that adds a specified amount
    of noise to the passed MultivariateNormal. This allows for adding known observational noise to test data.

    .. note::
        This likelihood can be used for exact or approximate inference.

    :param noise: Known observation noise (variance) for each training example.
    :type noise: torch.Tensor (... x N)
    :param learn_additional_noise: Set to true if you additionally want to
        learn added diagonal noise, similar to GaussianLikelihood.
    :type learn_additional_noise: bool, optional
    :param batch_shape: The batch shape of the learned noise parameter (default
        []) if :obj:`learn_additional_noise=True`.
    :type batch_shape: torch.Size, optional

    :var torch.Tensor noise: :math:`\sigma^2` parameter (noise)

    .. note::
        FixedNoiseGaussianLikelihood has an analytic marginal distribution.

    Example:
        >>> train_x = torch.randn(55, 2)
        >>> noises = torch.ones(55) * 0.01
        >>> likelihood = FixedNoiseGaussianLikelihood(noise=noises, learn_additional_noise=True)
        >>> pred_y = likelihood(gp_model(train_x))
        >>>
        >>> test_x = torch.randn(21, 2)
        >>> test_noises = torch.ones(21) * 0.02
        >>> pred_y = likelihood(gp_model(test_x), noise=test_noises)
    """

    def __init__(
        self,
        noise: torch.Tensor,
        learn_additional_noise: Optional[bool] = False,
        learn_noise_scale: Optional[bool] = False,
        batch_shape: Optional[torch.Size] = torch.Size(),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            noise_covar=FixedGaussianNoise(
                noise=noise, learn_noise_scale=learn_noise_scale
            )
        )

        # super().__init__(noise_covar=FixedGaussianNoise(noise=noise))
        self.second_noise_covar: Optional[HomoskedasticNoise] = None
        if learn_additional_noise:
            noise_prior = kwargs.get("noise_prior", None)
            noise_constraint = kwargs.get("noise_constraint", None)
            self.second_noise_covar = HomoskedasticNoise(
                noise_prior=noise_prior,
                noise_constraint=noise_constraint,
                batch_shape=batch_shape,
            )

    @property
    def noise(self) -> torch.Tensor:
        return self.noise_covar.noise + self.second_noise

    @noise.setter
    def noise(self, value: torch.Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def second_noise(self) -> Union[float, torch.Tensor]:
        if self.second_noise_covar is None:
            return 0.0
        else:
            return self.second_noise_covar.noise

    @second_noise.setter
    def second_noise(self, value: torch.Tensor) -> None:
        if self.second_noise_covar is None:
            raise RuntimeError(
                "Attempting to set secondary learned noise for FixedNoiseGaussianLikelihood, "
                "but learn_additional_noise must have been False!"
            )
        self.second_noise_covar.initialize(noise=value)

    def get_fantasy_likelihood(self, **kwargs: Any) -> "FixedNoiseGaussianLikelihood":
        if "noise" not in kwargs:
            raise RuntimeError(
                "FixedNoiseGaussianLikelihood.fantasize requires a `noise` kwarg"
            )

        old_noise_covar = self.noise_covar
        self.noise_covar = None
        fantasy_liklihood = deepcopy(self)
        self.noise_covar = old_noise_covar

        old_noise = old_noise_covar.noise
        new_noise = kwargs.get("noise")

        if old_noise.ndim == 1 and new_noise.ndim == 1:
            cat_noise = torch.cat([old_noise, new_noise], -1)
        elif old_noise.ndim == 2 and new_noise.ndim == 2:
            cat_noise = torch.block_diag(old_noise, new_noise)
        else:
            raise ValueError(
                "Noise tensors must have the same dimensions (both 1D or both 2D)."
            )

        fantasy_liklihood.noise_covar = FixedGaussianNoise(noise=cat_noise)
        return fantasy_liklihood

    def _shaped_noise_covar(
        self, base_shape: torch.Size, *params: Any, **kwargs: Any
    ) -> Union[torch.Tensor, LinearOperator]:
        if len(params) > 0:
            # we can infer the shape from the params
            shape = None
        else:
            # here shape[:-1] is the batch shape requested, and shape[-1] is `n`, the number of points
            shape = base_shape

        res = self.noise_covar(*params, shape=shape, **kwargs)

        if self.second_noise_covar is not None:
            res = res + self.second_noise_covar(*params, shape=shape, **kwargs)
        elif isinstance(res, ZeroLazyTensor):
            warnings.warn(
                "You have passed data through a FixedNoiseGaussianLikelihood that did not match the size "
                "of the fixed noise, *and* you did not specify noise. This is treated as a no-op.",
                warnings.GPInputWarning,
            )

        return res

    def marginal(
        self, function_dist: MultivariateNormal, *args: Any, **kwargs: Any
    ) -> MultivariateNormal:
        r"""
        :return: Analytic marginal :math:`p(\mathbf y)`.
        """
        return super().marginal(function_dist, *args, **kwargs)
