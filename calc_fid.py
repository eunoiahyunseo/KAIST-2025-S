import torch
import torch.nn as nn
from einops import rearrange
import torch_fidelity

class GenerativeModelWrapperForFID(torch_fidelity.GenerativeModelBase):
    def __init__(self, solver, batch_size, generation_steps, mask_token_id, device):
        super().__init__()
        self.solver = solver
        self.batch_size = batch_size
        self.generation_steps = generation_steps
        self.mask_token_id = mask_token_id
        self.device = device
        self.eval()

    @property
    def num_classes(self):
        return 10

    @property
    def z_size(self):
        return 128

    @property
    def z_type(self):
        return 'normal'

    # --- [FIXED] forward 메소드가 'labels' 인자를 받도록 수정 ---
    # torch-fidelity는 z와 labels, 두 개의 인자를 전달합니다.
    # 우리 모델은 unconditional하므로 labels를 무시하지만, 함수 시그니처는 맞춰줘야 합니다.
    def forward(self, z, labels):
        current_batch_size = z.size(0)
        x0_mask = torch.full((current_batch_size, 3, 32, 32), self.mask_token_id, device=self.device, dtype=torch.long)
        
        # Solver는 (B, S)를 반환
        generated_sequence = self.solver.sample(x_init=x0_mask, step_size=1 / self.generation_steps)
        
        # (B, C, H, W)로 변환
        generated_images = rearrange(generated_sequence, 'b (c h w) -> b c h w', c=3, h=32, w=32)
        
        return generated_images.to(torch.uint8)
    # --- End of Fix ---

def calculate_fid(solver, num_samples, batch_size, generation_steps, mask_token_id, device):
    print("Calculating FID against cifar10-val dataset...")
    
    generative_model = GenerativeModelWrapperForFID(
        solver=solver,
        batch_size=batch_size,
        generation_steps=generation_steps,
        mask_token_id=mask_token_id,
        device=device
    )

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=generative_model,
        input1_model_num_samples=num_samples,
        input2='cifar10-val',
        cuda=True,
        isc=False,
        fid=True,
        kid=False,
        verbose=False,
        input1_batch_size=batch_size,
    )
    
    return metrics_dict['frechet_inception_distance']