import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def generate_sbm_dynamic(M, block_sizes, state_prob_matrix):
    """
    SBM 규칙에 따라 그래프를 동적으로 생성합니다.

    Args:
        M (int): 그래프의 전체 노드 수.
        block_sizes (list of int): 각 블록에 속한 노드의 수. sum(block_sizes)는 M과 같아야 함.
        state_prob_matrix (np.ndarray): 상태 확률 분포 행렬.
                                        Shape: (k, k, N) 여기서 k는 블록 수, N은 상태 수.
                                        P[i, j]는 i블록과 j블록 간 엣지의 상태 분포 벡터.

    Returns:
        networkx.Graph: 생성된 그래프 객체.
    """
    # 입력 값 검증
    if sum(block_sizes) != M:
        raise ValueError("블록 크기의 합이 전체 노드 수 M과 일치해야 합니다.")
    
    num_blocks = len(block_sizes)
    num_states = state_prob_matrix.shape[2]
    states = np.arange(num_states) # [0, 1, ..., N-1]

    # 그래프 객체 초기화
    G = nx.Graph()
    G.add_nodes_from(range(M))

    # 각 노드가 어떤 블록에 속하는지 미리 매핑 (효율성)
    node_to_block_map = {}
    current_node_idx = 0
    for block_id, size in enumerate(block_sizes):
        for _ in range(size):
            node_to_block_map[current_node_idx] = block_id
            current_node_idx += 1
            
    # 동적 생성 시작 (노드를 하나씩 추가하며 연결)
    for i in range(M):
        for j in range(i):  # 이미 추가된 노드 j에 대해 연결 시도
            # 1. 두 노드의 블록 ID 확인
            block_i = node_to_block_map[i]
            block_j = node_to_block_map[j]

            # 2. 해당하는 상태 확률 분포 가져오기
            prob_dist = state_prob_matrix[block_i, block_j]

            # 3. 분포에 따라 엣지의 상태 샘플링
            sampled_state = np.random.choice(states, p=prob_dist)

            # 4. 상태가 0 (엣지 없음)이 아니면 그래프에 엣지 추가
            if sampled_state != 0:
                G.add_edge(i, j, state=sampled_state)
    
    return G

# --- 실행 예시 ---
if __name__ == '__main__':
    # 1. 파라미터 설정
    M = 100  # 전체 노드 수
    N = 3    # 상태 수 (0: 없음, 1: 강한 연결, 2: 약한 연결)
    k = 2    # 블록(커뮤니티) 수

    # 각 블록의 크기 설정 (50 + 50 = 100 = M)
    block_sizes = [50, 50]

    # 2. 상태 확률 분포 행렬(P) 정의 (k x k x N)
    #    각 행의 합은 1이 되어야 함
    P = np.zeros((k, k, N))

    # 블록 0 내부 (P_00): 강한 연결(상태1) 확률 높음
    P[0, 0] = [0.2, 0.7, 0.1]  # P(없음)=20%, P(강)=70%, P(약)=10%
    
    # 블록 1 내부 (P_11): 강한 연결(상태1) 확률 높음
    P[1, 1] = [0.3, 0.6, 0.1]  # P(없음)=30%, P(강)=60%, P(약)=10%

    # 블록 0과 1 사이 (P_01, P_10): 약한 연결(상태2) 확률 높음, 대부분 연결 없음
    P[0, 1] = [0.9, 0.05, 0.05] # P(없음)=90%, P(강)=5%, P(약)=5%
    P[1, 0] = P[0, 1]           # 대칭 행렬

    # 3. 그래프 생성 함수 호출
    print("SBM 그래프를 동적으로 생성합니다...")
    sbm_graph = generate_sbm_dynamic(M, block_sizes, P)
    print("그래프 생성 완료!")
    # print(nx.info(sbm_graph))
    print(sbm_graph)

    # 4. 생성된 그래프 시각화
    print("그래프 시각화...")
    pos = nx.spring_layout(sbm_graph, seed=42)
    
    # 노드 색상: 블록별로 다르게 지정
    node_colors = ['#1f78b4'] * block_sizes[0] + ['#33a02c'] * block_sizes[1]
    
    # 엣지 색상: 상태별로 다르게 지정
    edge_colors = []
    for u, v, data in sbm_graph.edges(data=True):
        if data['state'] == 1: # 강한 연결
            edge_colors.append('black')
        elif data['state'] == 2: # 약한 연결
            edge_colors.append('gray')

    plt.figure(figsize=(12, 12))
    nx.draw(sbm_graph, pos, node_color=node_colors, edge_color=edge_colors, with_labels=False, node_size=50)
    plt.title("Dynamically Generated SBM Graph")
    plt.show()