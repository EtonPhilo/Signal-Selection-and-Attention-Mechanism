import numpy as np
from numba import njit, prange

# -------------------------------------------------------------------
# 1) 单次模拟的 JIT 函数
# -------------------------------------------------------------------
@njit
def simulate_once_jit(alpha: float,
                      beta: float,
                      n_rounds: int,
                      seed: int) -> np.ndarray:
    """
    使用 Numba 编译的单次仿真函数：
      - 2 states, 3 signals, 2 features, 2 actions
      - urn-learning，初始权重都为 1
      - 成功 (act==state) 时才 +1
    返回长度 D=16 的扁平权重向量。
    """
    # 重设随机种子（Numba 内部使用 np.random）
    np.random.seed(seed)

    # urn 权重数组化
    sender_urn = np.ones((2, 3), dtype=np.float64)
    attention_urn = np.ones(2,      dtype=np.float64)
    receiver_action_urn = np.ones((2, 2, 2), dtype=np.float64)

    # 临时变量
    D0 = 2*3 + 2 + 2*2*2  # =16

    for _ in range(n_rounds):
        # 1) 随机 state
        state = np.random.randint(0, 2)

        # 2) 按权重抽 signal
        wsig = sender_urn[state]
        cum = np.cumsum(wsig)
        u = np.random.random() * cum[-1]
        sig = 0
        while cum[sig] < u:
            sig += 1

        # 3) signal → features
        if sig == 0:
            f0, f1 = 0, 0
        elif sig == 1:
            f0 = 1 if np.random.random() < alpha else 0
            f1 = 1 if np.random.random() < beta  else 0
        else:
            f0 = 1 if np.random.random() < alpha + 0.2 else 0 # consider difference
            f1 = np.random.randint(0, 2)

        # 4) 按注意力权重抽 feature
        cumA = np.cumsum(attention_urn)
        u = np.random.random() * cumA[-1]
        f_choice = 0 if u < cumA[0] else 1
        obs = f0 if f_choice == 0 else f1

        # 5) 按行为权重抽 action
        wact = receiver_action_urn[f_choice, obs]
        cumB = np.cumsum(wact)
        u = np.random.random() * cumB[-1]
        act = 0 if u < cumB[0] else 1

        # 6) 强化更新
        if act == state:
            sender_urn[state, sig]               += 1.0
            attention_urn[f_choice]              += 1.0
            receiver_action_urn[f_choice, obs, act] += 1.0

    # 扁平化输出
    out = np.empty(D0, dtype=np.float64)
    idx = 0
    # state0→3sig
    for j in range(3):
        out[idx] = sender_urn[0, j]; idx += 1
    # state1→3sig
    for j in range(3):
        out[idx] = sender_urn[1, j]; idx += 1
    # attention(2)
    for j in range(2):
        out[idx] = attention_urn[j]; idx += 1
    # action f0(obs0→2,obs1→2) + f1(...)
    for f in range(2):
        for obs in range(2):
            for a in range(2):
                out[idx] = receiver_action_urn[f, obs, a]
                idx += 1
    return out


# -------------------------------------------------------------------
# 2) 在 [0.5,1]^2 网格上并行跑仿真
# -------------------------------------------------------------------
@njit(parallel=True)
def simulate_grid_jit(alpha_grid: np.ndarray,
                      beta_grid:  np.ndarray,
                      n_rounds:   int,
                      n_reps:     int) -> np.ndarray:
    """
    并行地对每个 (i,j) 网格点下的 (alpha,beta) 做 n_reps 次 simulate_once_jit，
    最后平均。返回 shape=(n,n,16) 的 mean_weights。
    """
    n = alpha_grid.shape[0]
    D = 16
    mean_weights = np.zeros((n, n, D), dtype=np.float64)

    for i in prange(n):
        for j in range(n):
            acc = np.zeros(D, dtype=np.float64)
            a = alpha_grid[i, j]
            b = beta_grid[i, j]
            # seed0 固定偏移，防止重复
            base_seed = i * n + j
            for r in range(n_reps):
                acc += simulate_once_jit(a, b, n_rounds, base_seed + r)
            mean_weights[i, j, :] = acc / n_reps

    return mean_weights


def main():
    # 网格与仿真参数
    n_element = 20       # 每维格点数
    n_rounds  = 1_000_0  # 单次 simulate 的迭代轮数
    n_reps    = 1000_0       # 每个 (alpha,beta) 重复次数

    # 构造 [0.5,1] 网格中心
    step = 0.5 / n_element
    mids = 0.5 + (np.arange(n_element) + 0.5) * step
    alpha_grid, beta_grid = np.meshgrid(mids, mids, indexing='ij')

    # JIT 调用：第一次调用会有一次编译延迟
    mean_weights = simulate_grid_jit(alpha_grid, beta_grid,
                                     n_rounds, n_reps)

    # 保存结果
    flat = mean_weights.reshape(-1, mean_weights.shape[-1])
    header = ",".join(f"w{k}" for k in range(flat.shape[1]))
    np.savetxt("multifeat_signal_numba.csv",
               flat, delimiter=",",
               header=header, comments="")

    print("Done. Data saved to multifeat_signal_numba.csv")

if __name__ == "__main__":
    main()
