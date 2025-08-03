import torch
from torch import Tensor
from typing import Union, Callable

class CorrectorScheduler:
    """
    논문 Appendix D, Equation 35에 따라 corrector 계수를 스케줄링합니다.
    αt = 1 + alpha_param * t^a * (1 - t)^b
    여기서 반환하는 값은 div_free (즉, 논문의 beta_t = alpha_t - 1) 입니다.
    """
    def __init__(self, alpha_param: float = 10.0, a: float = 0.25, b: float = 0.25):
        self.alpha_param = alpha_param # 논문의 α (alpha parameter)
        self.a = a # 논문의 a (power for t)
        self.b = b # 논문의 b (power for (1-t))

    def __call__(self, t: Tensor):
        # 논문 αt = 1 + α * t^a * (1 - t)^b
        # 코드 div_free_t (beta_t) = αt - 1
        # 따라서 div_free_t = alpha_param * t^a * (1 - t)^b
        
        # 안정성을 위해 t와 (1-t)를 클램프
        t_safe = t.clamp(min=1e-8, max=1.0 - 1e-8)
        
        # 논문의 형태를 따라 계수 계산
        # alpha_t_val = 1.0 + self.alpha_param * (t_safe ** self.a) * ((1.0 - t_safe) ** self.b)
        # return alpha_t_val - 1.0 # beta_t = alpha_t - 1

        # 직접 div_free_t (beta_t) 계산
        div_free_t_val = self.alpha_param * (t_safe ** self.a) * ((1.0 - t_safe) ** self.b)
        return div_free_t_val