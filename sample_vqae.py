#!/usr/bin/env python3
"""
VQ-AE 모델에서 체크포인트를 불러와 샘플링하는 스크립트

사용법:
    python sample_vqae.py --checkpoint vqae_model.pth --num_samples 10 --seed 42
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from encoder_decoder import StrokeFusionVQAE
import argparse
import os

def load_and_sample(checkpoint_path='vqae_model.pth', num_samples=10, random_seed=42, output_dir='samples'):
    """
    저장된 체크포인트를 불러와서 랜덤 샘플링을 수행합니다.
    
    Args:
        checkpoint_path (str): 저장된 모델의 경로
        num_samples (int): 생성할 샘플의 개수
        random_seed (int): 재현 가능한 결과를 위한 시드
        output_dir (str): 결과를 저장할 디렉토리
    """
    print(f"\n=== VQ-AE 체크포인트 불러와서 샘플링 시작 ===")
    print(f"체크포인트 경로: {checkpoint_path}")
    print(f"생성할 샘플 수: {num_samples}")
    print(f"출력 디렉토리: {output_dir}")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 시드 설정
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # 하이퍼파라미터 (train_vqae.py와 동일하게 설정)
    MAX_STROKE_LEN = 24
    D_F = 256
    D_MODEL = 128
    D_IMG = 128
    D_SEQ = 64
    NUM_EMBEDDINGS = 256
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 중인 디바이스: {device}")
    
    # 모델 초기화
    model = StrokeFusionVQAE(
        max_stroke_len=MAX_STROKE_LEN,
        d_f=D_F,
        num_embeddings=NUM_EMBEDDINGS,
        d_seq=D_SEQ,
        d_img=D_IMG,
        d_model=D_MODEL
    ).to(device)
    
    # 체크포인트 불러오기
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ 체크포인트를 성공적으로 불러왔습니다: {checkpoint_path}")
    except FileNotFoundError:
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        return
    except Exception as e:
        print(f"❌ 체크포인트 불러오기 실패: {e}")
        return
    
    model.eval()
    
    with torch.no_grad():
        # 1. 랜덤 잠재 벡터 생성 (코드북에서 랜덤 인덱스 선택)
        random_indices = torch.randint(0, NUM_EMBEDDINGS, (num_samples,), device=device)
        random_encodings = torch.zeros(num_samples, NUM_EMBEDDINGS, device=device)
        random_encodings.scatter_(1, random_indices.unsqueeze(1), 1)
        
        # 코드북에서 해당하는 벡터들 가져오기
        z_q_random = torch.matmul(random_encodings, model.quantizer.embedding.weight)
        
        print(f"랜덤 잠재 벡터 생성 완료: {z_q_random.shape}")
        print(f"사용된 코드북 인덱스: {random_indices.cpu().numpy()}")
        
        # 2. 랜덤 샘플 생성
        print("스트로크 생성 중...")
        generated_coords, generated_pen_logits = model.generate(z_q_random)
        
        print("UDF 이미지 생성 중...")
        generated_udf = model.decoder.image_decoder(z_q_random)
        
        # 3. 결과 시각화 및 저장
        print(f"결과를 이미지 파일로 저장합니다 ({num_samples}개 샘플)...")
        
        all_coords = []
        all_udfs = []
        
        for i in range(num_samples):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'Generated Sample {i+1} (CodeBook Index: {random_indices[i].item()})', fontsize=14)
            
            # 펜 상태를 기반으로 유효한 길이 계산
            pen_states = torch.sigmoid(generated_pen_logits[i]) > 0.5
            if pen_states.sum() > 0:
                # 마지막으로 펜이 내려간 위치까지만 사용
                last_pen_down = torch.where(pen_states)[0][-1].item() + 1
                valid_length = min(last_pen_down, MAX_STROKE_LEN)
            else:
                valid_length = MAX_STROKE_LEN
            
            # --- 1. 생성된 스트로크 ---
            generated_stroke = generated_coords[i, :valid_length, :].cpu().numpy()
            all_coords.append(generated_stroke)
            
            axes[0].plot(generated_stroke[:, 0], generated_stroke[:, 1], 'g-', lw=2, marker='o', markersize=2)
            axes[0].set_title('Generated Stroke')
            axes[0].set_xlim(0, 1)
            axes[0].set_ylim(0, 1)
            axes[0].invert_yaxis()  # QuickDraw 스타일
            axes[0].set_aspect('equal')
            axes[0].grid(True, alpha=0.3)
            
            # 시작점과 끝점 표시
            if len(generated_stroke) > 0:
                axes[0].plot(generated_stroke[0, 0], generated_stroke[0, 1], 'go', markersize=8, label='Start')
                axes[0].plot(generated_stroke[-1, 0], generated_stroke[-1, 1], 'ro', markersize=8, label='End')
                axes[0].legend()
            
            # --- 2. 생성된 UDF ---
            generated_udf_np = generated_udf[i].squeeze().cpu().numpy()
            all_udfs.append(generated_udf_np)
            
            im = axes[1].imshow(generated_udf_np, cmap='hot')
            axes[1].set_title('Generated UDF')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], shrink=0.8)
            
            save_path = os.path.join(output_dir, f'generated_sample_{i+1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Generated Sample {i+1}이 '{save_path}'에 저장되었습니다.")
        
        # 4. 모든 샘플을 한 번에 보여주는 그리드 생성
        if num_samples > 1:
            cols = min(5, num_samples)
            rows = (num_samples + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            if rows == 1:
                axes = [axes] if cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i in range(num_samples):
                ax = axes[i] if num_samples > 1 else axes
                ax.plot(all_coords[i][:, 0], all_coords[i][:, 1], 'b-', lw=2)
                ax.set_title(f'Sample {i+1}\n(Code: {random_indices[i].item()})')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.invert_yaxis()
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
            
            # 빈 subplot 숨기기
            for i in range(num_samples, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            grid_save_path = os.path.join(output_dir, 'generated_samples_grid.png')
            plt.savefig(grid_save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"전체 샘플 그리드가 '{grid_save_path}'에 저장되었습니다.")
        
        # 5. 생성 결과를 numpy 파일로도 저장
        results = {
            'generated_coords': generated_coords.cpu().numpy(),
            'generated_pen_logits': generated_pen_logits.cpu().numpy(),
            'generated_udf': generated_udf.cpu().numpy(),
            'codebook_indices': random_indices.cpu().numpy(),
            'random_seed': random_seed,
            'num_samples': num_samples
        }
        
        results_path = os.path.join(output_dir, 'generated_results.npz')
        np.savez_compressed(results_path, **results)
        print(f"생성 결과가 '{results_path}'에 저장되었습니다.")
        
        # 6. 생성된 샘플들의 통계 정보 출력
        print(f"\n=== 생성 결과 통계 ===")
        print(f"생성된 샘플 수: {num_samples}")
        coord_lengths = [len(coords) for coords in all_coords]
        print(f"평균 스트로크 길이: {np.mean(coord_lengths):.1f} ± {np.std(coord_lengths):.1f}")
        print(f"최소/최대 스트로크 길이: {min(coord_lengths)} / {max(coord_lengths)}")
        print(f"사용된 코드북 크기: {NUM_EMBEDDINGS}")
        print(f"고유한 코드북 인덱스 수: {len(set(random_indices.cpu().numpy()))}")
        
        # UDF 이미지 통계
        udf_means = [np.mean(udf) for udf in all_udfs]
        print(f"UDF 평균값: {np.mean(udf_means):.4f} ± {np.std(udf_means):.4f}")
        
        print(f"\n=== 랜덤 샘플링 완료 ===")
        print(f"모든 결과가 '{output_dir}' 디렉토리에 저장되었습니다.")

def interpolate_samples(checkpoint_path, code_idx1, code_idx2, num_interpolation_steps=5, output_dir='interpolations'):
    """
    두 코드북 인덱스 사이를 보간하여 샘플들을 생성합니다.
    """
    print(f"\n=== 코드북 보간 샘플링 시작 ===")
    print(f"코드북 인덱스 {code_idx1} → {code_idx2}")
    print(f"보간 단계: {num_interpolation_steps}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 하이퍼파라미터
    MAX_STROKE_LEN = 24
    D_F = 256
    D_MODEL = 128
    D_IMG = 128
    D_SEQ = 64
    NUM_EMBEDDINGS = 256
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 모델 초기화 및 로드
    model = StrokeFusionVQAE(
        max_stroke_len=MAX_STROKE_LEN,
        d_f=D_F,
        num_embeddings=NUM_EMBEDDINGS,
        d_seq=D_SEQ,
        d_img=D_IMG,
        d_model=D_MODEL
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        # 시작과 끝 코드북 벡터 가져오기
        code_vec1 = model.quantizer.embedding.weight[code_idx1]
        code_vec2 = model.quantizer.embedding.weight[code_idx2]
        
        # 보간 벡터들 생성
        alphas = torch.linspace(0, 1, num_interpolation_steps, device=device)
        interpolated_vecs = []
        
        for alpha in alphas:
            interpolated_vec = (1 - alpha) * code_vec1 + alpha * code_vec2
            interpolated_vecs.append(interpolated_vec)
        
        z_q_interpolated = torch.stack(interpolated_vecs)
        
        # 보간된 샘플들 생성
        generated_coords, generated_pen_logits = model.generate(z_q_interpolated)
        generated_udf = model.decoder.image_decoder(z_q_interpolated)
        
        # 결과 시각화
        fig, axes = plt.subplots(2, num_interpolation_steps, figsize=(4*num_interpolation_steps, 8))
        
        for i in range(num_interpolation_steps):
            alpha = alphas[i].item()
            
            # 스트로크 시각화
            pen_states = torch.sigmoid(generated_pen_logits[i]) > 0.5
            if pen_states.sum() > 0:
                last_pen_down = torch.where(pen_states)[0][-1].item() + 1
                valid_length = min(last_pen_down, MAX_STROKE_LEN)
            else:
                valid_length = MAX_STROKE_LEN
            
            stroke = generated_coords[i, :valid_length, :].cpu().numpy()
            
            axes[0, i].plot(stroke[:, 0], stroke[:, 1], 'b-', lw=2)
            axes[0, i].set_title(f'α={alpha:.2f}')
            axes[0, i].set_xlim(0, 1)
            axes[0, i].set_ylim(0, 1)
            axes[0, i].invert_yaxis()
            axes[0, i].set_aspect('equal')
            axes[0, i].grid(True, alpha=0.3)
            
            # UDF 시각화
            udf = generated_udf[i].squeeze().cpu().numpy()
            im = axes[1, i].imshow(udf, cmap='hot')
            axes[1, i].set_title(f'UDF α={alpha:.2f}')
            axes[1, i].axis('off')
        
        plt.suptitle(f'Interpolation between Code {code_idx1} and Code {code_idx2}')
        plt.tight_layout()
        
        interp_save_path = os.path.join(output_dir, f'interpolation_{code_idx1}_{code_idx2}.png')
        plt.savefig(interp_save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"보간 결과가 '{interp_save_path}'에 저장되었습니다.")

def main():
    parser = argparse.ArgumentParser(description='VQ-AE 모델에서 샘플 생성')
    parser.add_argument('--checkpoint', '-c', type=str, default='vqae_model.pth',
                        help='모델 체크포인트 경로 (기본값: vqae_model.pth)')
    parser.add_argument('--num_samples', '-n', type=int, default=10,
                        help='생성할 샘플 개수 (기본값: 10)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='랜덤 시드 (기본값: 42)')
    parser.add_argument('--output_dir', '-o', type=str, default='samples',
                        help='결과 저장 디렉토리 (기본값: samples)')
    parser.add_argument('--interpolate', action='store_true',
                        help='보간 샘플링 모드')
    parser.add_argument('--code_idx1', type=int, default=0,
                        help='보간 시작 코드북 인덱스 (기본값: 0)')
    parser.add_argument('--code_idx2', type=int, default=255,
                        help='보간 끝 코드북 인덱스 (기본값: 255)')
    parser.add_argument('--interp_steps', type=int, default=5,
                        help='보간 단계 수 (기본값: 5)')
    
    args = parser.parse_args()
    
    # 체크포인트 파일 존재 확인
    if not os.path.exists(args.checkpoint):
        print(f"❌ 체크포인트 파일이 없습니다: {args.checkpoint}")
        print("먼저 train_vqae.py를 실행하여 모델을 훈련해주세요.")
        return
    
    if args.interpolate:
        # 보간 샘플링 실행
        interpolate_samples(
            checkpoint_path=args.checkpoint,
            code_idx1=args.code_idx1,
            code_idx2=args.code_idx2,
            num_interpolation_steps=args.interp_steps,
            output_dir=args.output_dir
        )
    else:
        # 일반 샘플링 실행
        load_and_sample(
            checkpoint_path=args.checkpoint,
            num_samples=args.num_samples,
            random_seed=args.seed,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main()
