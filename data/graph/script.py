import numpy as np
import os
import json

def npz_to_bin(npz_file, output_dir, split_ratio=0.9):
    """
    NPZ 파일의 그래프 데이터를 1차원으로 flatten하여 train.bin, val.bin으로 저장
    
    Args:
        npz_file: 입력 NPZ 파일 경로
        output_dir: 출력 디렉토리
        split_ratio: 훈련/검증 분할 비율
    """
    
    # 1. NPZ 파일 로드
    data = np.load(npz_file)
    print("Loading data from:", npz_file)
    print("Available keys:", list(data.keys()))
    
    # 2. 모든 배열을 1차원으로 flatten하고 연결
    all_tokens = []
    
    for key in data.keys():
        arr = data[key]
        print(f"Processing {key}: shape={arr.shape}, dtype={arr.dtype}")
        
        # 각 배열을 1차원으로 flatten
        flattened = arr.flatten()
        
        # dtype을 uint16으로 변환 (필요한 경우)
        if flattened.dtype != np.uint16:
            # 값의 범위 확인
            min_val, max_val = flattened.min(), flattened.max()
            print(f"  Value range: {min_val} to {max_val}")
            
            # uint16 범위에 맞게 조정 (0-65535)
            if min_val < 0 or max_val > 65535:
                # 정규화 또는 클리핑 필요
                if min_val < 0:
                    flattened = flattened - min_val  # 최소값을 0으로
                if flattened.max() > 65535:
                    flattened = (flattened / flattened.max() * 65535).astype(np.uint16)
                else:
                    flattened = flattened.astype(np.uint16)
            else:
                flattened = flattened.astype(np.uint16)
        
        all_tokens.extend(flattened)
        print(f"  Added {len(flattened)} tokens")
    
    # 3. 전체 토큰 배열 생성
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    print(f"Total tokens: {len(all_tokens)}")
    
    # 4. 훈련/검증 데이터 분할
    split_idx = int(len(all_tokens) * split_ratio)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]
    
    # 5. 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 6. bin 파일로 저장
    train_path = os.path.join(output_dir, 'train.bin')
    val_path = os.path.join(output_dir, 'val.bin')
    
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)
    
    print(f"Train data: {len(train_tokens)} tokens -> {train_path}")
    print(f"Val data: {len(val_tokens)} tokens -> {val_path}")
    
    # 7. 메타데이터 저장
    vocab_size = int(all_tokens.max()) + 1  # 0부터 시작하므로 +1
    
    meta = {
        'vocab_size': vocab_size,
        'total_tokens': len(all_tokens),
        'train_tokens': len(train_tokens),
        'val_tokens': len(val_tokens),
        'original_keys': list(data.keys()),
        'value_range': [int(all_tokens.min()), int(all_tokens.max())]
    }
    
    meta_path = os.path.join(output_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"Metadata saved to: {meta_path}")
    print(f"Vocabulary size: {vocab_size}")
    
    return train_tokens, val_tokens, meta

# 사용 예시
if __name__ == "__main__":
    # NPZ 파일을 bin 파일로 변환
    train_tokens, val_tokens, meta = npz_to_bin(
        '/home/hyunseo/workspace/KAIST-2025-S/discrete_model_testbed/data/graph/graph_dataset.npz', 
        '/home/hyunseo/workspace/KAIST-2025-S/discrete_model_testbed/data/graph/',
        split_ratio=0.8
    )
    
    # 결과 검증
    print("\n=== Verification ===")
    train_data = np.memmap('data/graph/train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap('data/graph/val.bin', dtype=np.uint16, mode='r')
    
    print(f"Train memmap shape: {train_data.shape}")
    print(f"Val memmap shape: {val_data.shape}")
    print(f"First 10 train tokens: {train_data[:10]}")
    print(f"Last 10 val tokens: {val_data[-10:]}")