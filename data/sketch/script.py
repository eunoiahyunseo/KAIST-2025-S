import os
import numpy as np
import random

# --- 설정 --- #
# 경로를 현재 폴더 기준으로 단순화했습니다. 실제 환경에 맞게 수정하세요.
input_npz_path = '/home/hyunseo/workspace/KAIST-2025-S/discrete_model_testbed/data/sketch/sketches.npz' 
output_dir = '/home/hyunseo/workspace/KAIST-2025-S/discrete_model_testbed/data/sketch'
# ---------------- #

def save_split():
    if not os.path.exists(input_npz_path):
        print(f"에러: '{input_npz_path}' 파일을 찾을 수 없습니다.")
        return
        
    print(f"'{input_npz_path}' 파일 로딩 중...")
    loaded_data = np.load(input_npz_path)
        
    # 키 리스트를 새로 만들어야 random.shuffle이 가능합니다.
    keys = list(loaded_data.files)
    random.shuffle(keys)
    
    n = len(keys)
    train_size = int(n * 0.8)
    
    train_keys = keys[:train_size]
    val_keys = keys[train_size:]
    print(f"키 분할 완료: train={len(train_keys)}개, val={len(val_keys)}개")
    
    
    train_dic = {key: loaded_data[key] for key in train_keys}
    val_dic = {key: loaded_data[key] for key in val_keys}
    
    train_output_path = os.path.join(output_dir, 'train.npz')
    val_output_path = os.path.join(output_dir, 'val.npz')
    
    print(f"'{train_output_path}' 파일 저장 중...")
    np.savez_compressed(train_output_path, **train_dic)
    
    print(f"'{val_output_path}' 파일 저장 중...")
    np.savez_compressed(val_output_path, **val_dic)
    
    print("\n저장 확인:")
    
    check_train_data = np.load(train_output_path)
    first_key = check_train_data.files[0]
    print(f"train.npz의 첫 번째 데이터 ('{first_key}') shape: {check_train_data[first_key].shape}")

    print("\n모든 작업이 완료되었습니다.")


if __name__ == '__main__':
    save_split()