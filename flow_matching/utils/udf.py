import numpy as np
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
import os
import ndjson


def distance_point_to_segment_squared(points, p1, p2):
    """
    그리드의 각 점에서 선분(p1, p2)까지의 최단 거리의 제곱을 계산합니다.
    이 계산은 벡터화되어 매우 효율적입니다.

    Args:
        points (np.ndarray): (H, W, 2) 형태의 그리드 점들.
        p1 (np.ndarray): (2,) 형태의 선분 시작점.
        p2 (np.ndarray): (2,) 형태의 선분 끝점.

    Returns:
        np.ndarray: (H, W) 형태의 거리 제곱 값 그리드.
    """
    # 선분 벡터와 길이 제곱 계산
    segment_vec = p2 - p1
    segment_length_sq = np.sum(segment_vec**2)

    # 선분이 점일 경우 (길이가 0)
    if segment_length_sq < 1e-9:
        return np.sum((points - p1)**2, axis=-1)

    # 그리드 점에서 선분 시작점까지의 벡터
    points_vec = points - p1

    # 그리드 점들을 선분 벡터에 투영(projection)
    # t는 선분 상의 상대적 위치 (0: p1, 1: p2)
    t = np.sum(points_vec * segment_vec, axis=-1) / segment_length_sq

    # t를 [0, 1] 범위로 제한하여 선분 내에만 있도록 함
    # t &lt; 0 이면 가장 가까운 점은 p1
    # t &gt; 1 이면 가장 가까운 점은 p2
    t_clamped = np.clip(t, 0, 1)

    # 각 그리드 점에 가장 가까운 선분 상의 점 계산
    # t_clamped를 (H, W, 1)로 확장하여 벡터 연산
    closest_points_on_segment = p1 + t_clamped[..., np.newaxis] * segment_vec

    # 그리드 점과 가장 가까운 점 사이의 거리 제곱 계산
    squared_distances = np.sum((points - closest_points_on_segment)**2, axis=-1)
    
    return squared_distances

def create_udf_from_stroke(stroke_points, resolution=64, gamma=50.0):
    """
    하나의 스트로크 데이터로부터 Unsigned Distance Function (UDF) 이미지를 생성합니다.
    
    Args:
        stroke_points (np.ndarray): (N, 2) 형태의 스트로크 점들의 시퀀스 (x, y 좌표).
                                    좌표는 [0, 1] 범위로 정규화되어야 합니다.
        resolution (int): 생성할 UDF 이미지의 해상도 (가로, 세로 동일).
        gamma (float): 지수 함수의 감쇠 계수. 클수록 UDF가 날카로워집니다.

    Returns:
        np.ndarray: (resolution, resolution) 형태의 UDF 이미지.
    """
    if stroke_points.shape[0] < 2:
        print("스트로크는 최소 2개 이상의 점이 필요합니다.")
        return np.zeros((resolution, resolution))

    # 1. 2D 그리드 생성
    grid_coords = np.linspace(0, 1, resolution)
    grid_x, grid_y = np.meshgrid(grid_coords, grid_coords)
    # (resolution, resolution, 2) 형태로 변환
    grid_points = np.stack([grid_x, grid_y], axis=-1)
    
    # 최종 UDF 필드를 저장할 배열 초기화
    # 모든 값이 0으로 시작하며, 각 선분의 영향을 받아 점차 업데이트됨
    final_udf = np.zeros((resolution, resolution))

    # 2. 스트로크의 모든 선분(segment)을 순회
    for i in range(len(stroke_points) - 1):
        p1 = stroke_points[i]
        p2 = stroke_points[i+1]
        
        # 3. 그리드와 현재 선분 간의 거리 제곱 계산
        dist_sq = distance_point_to_segment_squared(grid_points, p1, p2)
        
        # 4. 지수 함수 적용 (논문의 식 (3))
        exp_dist = np.exp(-gamma * dist_sq)
        
        # 5. 최대값 집계 (논문의 식 (4))
        # 현재까지 계산된 UDF와 새로 계산된 지수 거리 중 더 큰 값을 선택
        final_udf = np.maximum(final_udf, exp_dist)
        
    return final_udf

if __name__ == '__main__':
    category = "apple"
    data_path = f"../../data/quickdraw/{category}.ndjson"

    with open(data_path, 'r') as f:
        data = ndjson.load(f)
        
    dic = {}
    cnt = 0
    for i in data: 
        if i['recognized'] is True:
            drawing = i['drawing']            
            dic[str(cnt)] = drawing  # 리스트 형태로 저장
            cnt += 1
            if cnt >= 10:
                break
            
    sample_stroke = np.array(dic['1'][0]).T / 255.0
    print('example_stroke: ', sample_stroke)

    # UDF 생성
    # 논문의 Figure 2처럼 gamma 값을 바꿔가며 테스트 가능
    udf_image_gamma_50 = create_udf_from_stroke(sample_stroke, resolution=64, gamma=50)
    udf_image_gamma_200 = create_udf_from_stroke(sample_stroke, resolution=64, gamma=200)

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 원본 스트로크
    axes[0].plot(sample_stroke[:, 0], sample_stroke[:, 1], marker='o', color='red')
    axes[0].set_title("Original Stroke")
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].invert_yaxis() # 이미지 좌표계와 동일하게 위쪽을 0으로

    # 2. UDF (gamma = 50)
    im1 = axes[1].imshow(udf_image_gamma_50, cmap='gray', origin='upper', extent=[0, 1, 1, 0])
    axes[1].set_title("UDF (gamma = 50)")
    fig.colorbar(im1, ax=axes[1])

    # 3. UDF (gamma = 200)
    im2 = axes[2].imshow(udf_image_gamma_200, cmap='gray', origin='upper', extent=[0, 1, 1, 0])
    axes[2].set_title("UDF (gamma = 200)")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig('plot.png')
    plt.show()