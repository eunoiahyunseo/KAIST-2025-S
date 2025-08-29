# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from math import ceil
from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch.nn import functional as F
from einops import rearrange

from flow_matching.path import MixtureDiscreteProbPath

from flow_matching.solver.solver import Solver
from flow_matching.utils import categorical, ModelWrapper
from .utils import get_nearest_times

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class MixtureDiscreteEulerSolver(Solver):
    r"""Solver that simulates the CTMC process :math:`(X_t)_{t_{\text{init}}\leq t\leq t_{\text{final}}}` defined by :math:`p_t` the marginal probability path of ``path``.
    Given :math:`X_t \sim p_t`, the algorithm of solver step from :math:`t` to :math:`t+h` for the i-th coordinate is:

    .. math::

        \begin{align*}
            & X_1^i \sim p_{1|t}^i(\cdot|X_t)\\
            & \lambda^i \gets \sum_{x^i\ne X_t^i} u_t^i(x^i, X_t^i|X_1^i)\\
            & Z^i_{\text{change}} \sim U[0,1]\\
            & X_{t+h}^i \sim \begin{cases}
                \frac{u_t^i(\cdot, X_t^i|X_1^i)}{\lambda^i}(1-\delta_{X_t^i}(\cdot)) \text{ if $Z^i_{\text{change}}\le 1-e^{-h\lambda^i}$}\\
                \delta_{X_t^i}(\cdot) \text{ else }
            \end{cases}
        \end{align*}

    Where :math:`p_{1|t}(\cdot|X_t)` is the output of ``model``, and the conditional probability velocity is of the mixture probability path is:

    .. math::

        u_t^i(x^i, y^i|x_1^i) = \hat{u}_t^i(x^i, y^i|x_1^i) + c_{\text{div\_free}}\left[\hat{u}_t^i(x^i, y^i|x_1^i) - \check{u}_t^i(x^i, y^i|x_1^i) \right],

    where

    .. math::
        \hat{u}_t^i(x^i, y^i|x_1^i) = \frac{\dot{\kappa}_t}{1-\kappa_t} \left[ \delta_{x_1^i}(x^i) - \delta_{y^i}(x^i) \right],

    and

    .. math::

        \check{u}_t^i(x^i, y^i|x_1^i) = \frac{\dot{\kappa}_t}{\kappa_t}\left[ \delta_{y^i}(x^i) - p(x^i) \right].

    The source distribution :math:`p(x^i)` is given by ``p``.

    Args:
        model (ModelWrapper): trained with x-prediction, outputting posterior probabilities (in the range :math:`[0,1]`), output must be [..., vocabulary_size].
        path (MixtureDiscreteProbPath): Probability path used for x-prediction training.
        vocabulary_size (int): size of the discrete vocabulary.
        source_distribution_p (Optional[Tensor], optional): Source distribution, must be of shape [vocabulary_size]. Required only when divergence-free term for the probability velocity is non-zero. Defaults to None.
    """

    def __init__(
        self,
        model: ModelWrapper,
        path: MixtureDiscreteProbPath,
        vocabulary_size: int,
        source_distribution_p: Optional[Tensor] = None,
        predictor = None,
    ):
        super().__init__()
        self.model = model
        self.path = path
        self.vocabulary_size = vocabulary_size

        if source_distribution_p is not None:
            assert source_distribution_p.shape == torch.Size(
                [vocabulary_size]
            ), f"Source distribution p dimension must match the vocabulary size {vocabulary_size}. Got {source_distribution_p.shape}."

        self.source_distribution_p = source_distribution_p

        self.predictor = predictor

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        step_size: Optional[float],
        div_free: Union[float, Callable[[float], float]] = 0.0,
        dtype_categorical: torch.dtype = torch.float32,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        verbose: bool = False,
        **model_extras,
    ) -> Tensor:
        """
        Sample a sequence of discrete values from the given model.

        .. code-block:: python

            import torch
            from flow_matching.utils import ModelWrapper
            from flow_matching.solver import MixtureDiscreteEulerSolver

            class DummyModel(ModelWrapper):
                def __init__(self):
                    super().__init__(None)
                def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
                    return ...

            model = DummyModel()
            solver = MixtureDiscreteEulerSolver(model=model)

            x_init = torch.LongTensor([122, 725])
            step_size = 0.001
            time_grid = torch.tensor([0.0, 1.0])

            result = solver.sample(x_init=x_init, step_size=step_size, time_grid=time_grid)

        Args:
            x_init (Tensor): The initial state.
            step_size (Optional[float]): If float then time discretization is uniform with the given step size. If None then time discretization is set to be time_grid.
            div_free (Union[float, Callable[[float], float]]): The coefficient of the divergence-free term in the probability velocity. Can be either a float or a time dependent function. Defaults to 0.0.
            dtype_categorical (torch.dtype): Precision to use for categorical sampler. Defaults to torch.float32.
            time_grid (Tensor): The CTMC process is solved in the interval [time_grid[0], time_grid[-1]] and if step_size is None then time discretization is set by the time grid. Defaults to torch.tensor([0.0,1.0]).
            return_intermediates (bool): If True then return intermediate time steps according to time_grid. Defaults to False.
            verbose (bool): Whether to print progress bars. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Tensor: The sampled sequence of discrete values.

        Raises:
            ImportError: To run in verbose mode, tqdm must be installed.
        """
        # print(self.source_distribution_p)
        if not div_free == 0.0:
            assert (
                self.source_distribution_p is not None
            ), "Source distribution p must be specified in order to add a divergence-free term to the probability velocity."

        # Initialize the current state `x_t` with the initial state `X_0`.
        time_grid = time_grid.to(device=x_init.device)
        is_image = x_init.dim() == 4

        if is_image:
            B, C, H, W = x_init.shape
            x_init = rearrange(x_init, 'b c h w -> b (c h w)')

        if step_size is None:
            # If step_size is None then set the t discretization to time_grid.
            t_discretization = time_grid
            n_steps = len(time_grid) - 1
        else:
            # If step_size is float then t discretization is uniform with step size set by step_size.
            t_init = time_grid[0].item()
            t_final = time_grid[-1].item()
            assert (
                t_final - t_init
            ) > step_size, f"Time interval [time_grid[0], time_grid[-1]] must be larger than step_size. Got a time interval [{t_init}, {t_final}] and step_size {step_size}."

            n_steps = ceil((t_final - t_init) / step_size)
            t_discretization = torch.tensor(
                [t_init + step_size * i for i in range(n_steps)] + [t_final],
                device=x_init.device,
            )


        x_t = x_init.clone()
        steps_counter = 0
        res = []

        if verbose:
            if not TQDM_AVAILABLE:
                raise ImportError(
                    "tqdm is required for verbose mode. Please install it."
                )
            ctx = tqdm(total=t_final, desc=f"NFE: {steps_counter}")
        else:
            ctx = nullcontext()

        with ctx:
            for i in range(n_steps):

                t = t_discretization[i : i + 1]
                h = t_discretization[i + 1 : i + 2] - t_discretization[i : i + 1]
                
                t_batch = t.repeat(x_t.shape[0])

                if is_image:
                    # U-Net은 4D 입력을 기대하므로, 1D 시퀀스를 4D 이미지로 다시 변환합니다.
                    model_input = rearrange(x_t, 'b (c h w) -> b c h w', c=C, h=H, w=W)
                else:
                    # 시퀀스 모델은 2D 입력을 그대로 사용합니다.
                    model_input = x_t

                # Sample x_1 ~ p_1|t( \cdot |x_t) # (B, S, V) <- softmax result
                p_1t_uncond = self.model(x_t=model_input, times=t_batch, **model_extras)
                # choose x_1 according to the categorical distribution
                # (B, S)
                x_1_uncond = categorical(p_1t_uncond.to(dtype=dtype_categorical))

                scheduler_output = self.path.scheduler(t=t)
                k_t, d_k_t = scheduler_output.alpha_t, scheduler_output.d_alpha_t

                # delta_1_uncond: (B, S, V)
                # means probability velocity 
                delta_1_uncond = F.one_hot(x_1_uncond, num_classes=self.vocabulary_size).to(k_t.dtype)
                
                epsilon = 1e-8
                k_t_safe = k_t.clamp(min=epsilon)
                one_minus_k_t_safe = (1.0 - k_t_safe).clamp(min=epsilon)

                forward_coeff = d_k_t / one_minus_k_t_safe
                # u의 각 요소 u(x_i)는 각 토큰이 x_i로 변하려는 순간적인 비율 (rate)를 의미한다.
                u_uncond = forward_coeff * delta_1_uncond # 
                u = u_uncond

                # Checks if final step
                if i == n_steps - 1:
                    x_t = x_1_uncond
                else:
            
                    # for the corrector_scheduler
                    div_free_t = div_free(t) if callable(div_free) else div_free

                    if div_free_t > 0:
                        p_0 = self.source_distribution_p[(None,) * x_t.dim()]
                        u = u + div_free_t * d_k_t / (k_t * (1 - k_t)) * (
                            (1 - k_t) * p_0 + k_t * delta_1_uncond
                        )

                    # Set u_t(x_t|x_t,x_1) = 0
                    delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size)
                    # 현재 토큰 x_t에 해당하는 위치의 u값을 0으로 만든다.
                    # 이렇게 하면 텐서는 순수하게 '들어오는' 양수 속도만 남게 된다. 음수 속도는 잠시 제거된다.
                    u = torch.where(
                        delta_t.to(dtype=torch.bool), torch.zeros_like(u), u
                    )

                    # Sample x_t ~ u_t( \cdot |x_t,x_1)
                    # 모든 들어오는 속도의 총합이 된다.
                    # CE에 따라 intensity = -u_t(x_t)
                    intensity = u.sum(dim=-1)  # Assuming u_t(xt|xt,x1) := 0
                    
                    # probability to jumpy another state
                    # 바꿀지 말지를 결정하는 확률적 로직
                    # Poisson process에 따라 비율(intensity) \lambda로 발생하는 사건이 작은 시간 h동안 한 번 이상 일어날 확률은
                    # 1 - exp(-\lambda h) 이다.
                    mask_jump = torch.rand(
                        size=x_t.shape, device=x_t.device
                    ) < 1 - torch.exp(-h * intensity)

                    # when you decided to jump, then jump according to Q(x, y) / \lambda(x)
                    if mask_jump.sum() > 0:
                        x_t[mask_jump] = categorical(
                            u[mask_jump].to(dtype=dtype_categorical)
                        )

                steps_counter += 1
                t = t + h

                if return_intermediates and (t in time_grid):
                    res.append(x_t.clone())

                if verbose:
                    ctx.n = t.item()
                    ctx.refresh()
                    ctx.set_description(f"NFE: {steps_counter}")

        return x_t
