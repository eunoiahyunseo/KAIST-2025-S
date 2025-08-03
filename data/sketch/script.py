import os
import numpy as np
import random
import json # json 모듈 추가

# --- 설정 --- #
# 경로를 현재 폴더 기준으로 단순화했습니다. 실제 환경에 맞게 수정하세요.
file_name = 'sketches_mountain.npz'
# 입력 파일과 출력 폴더 경로를 명확히 했습니다.
input_npz_path = f'/home/hyunseo/workspace/KAIST-2025-S/discrete_model_testbed/data/sketch/{file_name}' 
output_dir = '/home/hyunseo/workspace/KAIST-2025-S/discrete_model_testbed/data/sketch'
# ---------------- #

def save_split():
    if not os.path.exists(input_npz_path):
        print(f"에러: '{input_npz_path}' 파일을 찾을 수 없습니다.")
        return
        
    print(f"'{input_npz_path}' 파일 로딩 중...")
    # allow_pickle=False는 보안상 권장됩니다.
    loaded_data = np.load(input_npz_path, allow_pickle=False)
    
    # ---  최대 토큰 값 찾기 및 meta.json 저장 로직 (추가된 부분) ---
    print("\n데이터셋 전체에서 최대 토큰 값 찾는 중...")
    max_token_value = 0
    # 모든 스케치(배열)를 순회하며 최대값을 찾습니다.
    for key in loaded_data.files:
        # np.max()는 배열 전체에서 최대값을 효율적으로 찾습니다.
        current_max = np.max(loaded_data[key])
        if current_max > max_token_value:
            max_token_value = current_max
    
    # 어휘사전 크기는 (최대 토큰 값 + 1) 입니다 (0부터 시작하므로).
    vocab_size = int(max_token_value) + 1
    
    print(f"찾은 최대 토큰 값: {int(max_token_value)}")
    print(f"계산된 vocab_size: {vocab_size}")
    
    # meta.json 파일 생성 및 저장
    meta_data = {'vocab_size': vocab_size}
    meta_output_path = os.path.join(output_dir, 'meta.json')
    
    print(f"'{meta_output_path}' 파일 저장 중...")
    with open(meta_output_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=4)
    # -------------------------------------------------------------
        
    # 키 리스트를 새로 만들어야 random.shuffle이 가능합니다.
    keys = list(loaded_data.files)
    random.shuffle(keys)
    
    n = len(keys)
    train_size = int(n * 0.8)
    
    train_keys = keys[:train_size]
    val_keys = keys[train_size:]
    print(f"\n키 분할 완료: train={len(train_keys)}개, val={len(val_keys)}개")
    
    
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
    # check_train_data에 파일이 있는지 확인
    if check_train_data.files:
        first_key = check_train_data.files[0]
        print(f"train.npz의 첫 번째 데이터 ('{first_key}') shape: {check_train_data[first_key].shape}")
    else:
        print("train.npz에 저장된 데이터가 없습니다.")

    print("\n모든 작업이 완료되었습니다.")


if __name__ == '__main__':
    save_split()