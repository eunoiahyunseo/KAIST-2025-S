import os
import numpy as np
import random
import json # json 모듈 추가
from tqdm import tqdm # tqdm 라이브러리 import


# --- 설정 --- #
# 경로를 현재 폴더 기준으로 단순화했습니다. 실제 환경에 맞게 수정하세요.
file_name = 'sketches_apple.npz'
vocab_size = 258
# 입력 파일과 출력 폴더 경로를 명확히 했습니다.
input_npz_path = f'/home/hyunseo/workspace/KAIST-2025-S/discrete_model_testbed/data/full_sketch/{file_name}' 
output_dir = '/home/hyunseo/workspace/KAIST-2025-S/discrete_model_testbed/data/full_sketch'
# ---------------- #

def save_split():
    if not os.path.exists(input_npz_path):
        print(f"에러: '{input_npz_path}' 파일을 찾을 수 없습니다.")
        return
        
    print(f"'{input_npz_path}' 파일 로딩 중...")
    loaded_data = np.load(input_npz_path, allow_pickle=False)
    meta_data = {'vocab_size': vocab_size}
    meta_output_path = os.path.join(output_dir, 'meta.json')
    
    print(f"'{meta_output_path}' 파일 저장 중...")
    with open(meta_output_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=4)
        
    keys = list(loaded_data.files)
    random.shuffle(keys)
    
    n = len(keys)
    train_size = int(n * 0.9)
    
    train_keys = keys[:train_size]
    val_keys = keys[train_size:]
    print(f"\n키 분할 완료: train={len(train_keys)}개, val={len(val_keys)}개")
    
    
    train_output_path = os.path.join(output_dir, 'train.npz')
    val_output_path = os.path.join(output_dir, 'val.npz')
    
    print(f"'{train_output_path}' 파일 저장 중 (메모리 효율적 방식)...")
    # tqdm으로 train_keys를 감싸서 진행률을 표시합니다.
    train_dic_generator = {key: loaded_data[key] for key in tqdm(train_keys, desc="Saving train data")}
    np.savez_compressed(train_output_path, **train_dic_generator)

    print(f"'{val_output_path}' 파일 저장 중 (메모리 효율적 방식)...")
    val_dic_generator = {key: loaded_data[key] for key in tqdm(val_keys, desc="Saving val data")}
    np.savez_compressed(val_output_path, **val_dic_generator)


    
    print("\n저장 확인:")
    
    check_train_data = np.load(train_output_path)
    # check_train_data에 파일이 있는지 확인
    if check_train_data.files:
        first_key = check_train_data.files[0]
        print(f"train.npz의 첫 번째 데이터 ('{first_key}') shape: {check_train_data[first_key].shape}")
    else:
        print("train.npz에 저장된 데이터가 없습니다.")

    print("\n모든 작업이 완료되었습니다.")
    print('check first data', loaded_data['0'])

if __name__ == '__main__':
    save_split()