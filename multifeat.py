import numpy as np
import random
from multiprocessing import Pool, cpu_count
from functools import partial

def simulate_once(alpha: float,
                  beta: float,
                  n_rounds: int,
                  seed: int) -> np.ndarray:
    """
    核心单次模拟逻辑：给定 (alpha, beta, n_rounds, seed)
    2 states, 3 signals, 2 actions, 2 binary features，
    返回长度 D=16 的权重向量。
    """
    rng = random.Random(seed)
    # 1) Sender urns: {state: [w_sig0,w_sig1,w_sig2]}
    sender_urn = {0: [1.0]*3, 1: [1.0]*3}
    # 2) Attention urn: 两个特征的权重
    attention_urn = [1.0, 1.0]
    # 3) Receiver action urns: 对每个特征 f∈{0,1}，每个 obs∈{0,1} 都有 [w_act0,w_act1]
    receiver_action_urn = [
        {0: [1.0,1.0], 1: [1.0,1.0]}
        for _ in range(2)
    ]

    for _ in range(n_rounds):
        # --- 1. draw state ---
        state = rng.choice([0, 1])
        # --- 2. sender emits one of 3 signals ---
        sig = rng.choices([0, 1, 2],
                          weights=sender_urn[state], k=1)[0]
        # --- 3. signal → (f0,f1) with success rates alpha,beta ---
        if sig == 0:
            f0, f1 = 0, 0
        elif sig == 1:
            f0 = 1 if rng.random() < alpha else 0
            f1 = 1 if rng.random() < beta  else 0
        else:  # sig == 2
            f0 = 1 if rng.random() < alpha else 0
            f1 = rng.choice([0, 1])
        # --- 4. receiver attends one feature ---
        f_choice = rng.choices([0, 1],
                               weights=attention_urn, k=1)[0]
        obs = (f0 if f_choice == 0 else f1)
        # --- 5. receiver acts ---
        act = rng.choices([0, 1],
                          weights=receiver_action_urn[f_choice][obs],
                          k=1)[0]
        # --- 6. 强化更新（成功才+1）---
        if act == state:
            sender_urn[state][sig]                += 1.0
            attention_urn[f_choice]               += 1.0
            receiver_action_urn[f_choice][obs][act] += 1.0

    # 扁平化输出：state0→3signal, state1→3signal, attention(2),
    # action f0:{obs0→2,obs1→2}, action f1:{obs0→2,obs1→2}
    out = []
    out += sender_urn[0] + sender_urn[1]
    out += attention_urn
    for f in (0, 1):
        out += receiver_action_urn[f][0]
        out += receiver_action_urn[f][1]
    return np.array(out, dtype=float)


def simulate_alpha_beta(alpha: float,
                        beta: float,
                        n_rounds: int,
                        n_reps: int,
                        seed0: int) -> np.ndarray:
    """
    将 n_reps 次 simulate_once 合并到一次调用中，并行任务粒度为 (alpha,beta)：
    seed0 用于生成 n_reps 个不同种子。
    最后返回对 n_reps 次结果的平均值向量。
    """
    D = 16  # simulate_once 输出长度
    acc = np.zeros(D, dtype=float)
    for k in range(n_reps):
        acc += simulate_once(alpha, beta, n_rounds, seed0 + k)
    return acc / n_reps


def grid_experiment_parallel(n_element: int,
                             n_rounds: int,
                             n_reps: int):
    """
    在 [0.5,1]×[0.5,1] 划 n_element×n_element 网格，
    对每个 (alpha,beta) 并行跑 n_reps 次模拟并平均。
    返回：
      alpha_grid, beta_grid: (n,n)
      mean_weights: (n,n,D)
    """
    # 1. 构造 [0.5,1] 的网格中心点
    step = 0.5 / n_element
    mids = 0.5 + (np.arange(n_element) + 0.5) * step
    alpha_grid, beta_grid = np.meshgrid(mids, mids, indexing='ij')

    # 2. 构造参数列表（每项是一个 (alpha,beta,n_rounds,n_reps,seed0)）
    tasks = []
    for i in range(n_element):
        for j in range(n_element):
            a = alpha_grid[i, j]
            b = beta_grid[i, j]
            # seed0 取 i*n_element+j 作为偏移量，可以防止不同 (a,b) 重复
            tasks.append((a, b, n_rounds, n_reps, i*n_element + j))

    # 3. 并行执行：使用 chunksize 减少任务分发开销
    n_workers = max(1, cpu_count() - 1)
    simulate_partial = simulate_alpha_beta  # 直接用，不需要再封装
    with Pool(n_workers) as pool:
        # 设置 chunksize= max(1, len(tasks)//(n_workers*4))
        chunksize = max(1, len(tasks) // (n_workers * 4))
        results = pool.starmap(simulate_partial, tasks, chunksize)

    # 4. reshape & 平均
    D = results[0].size
    arr = np.array(results)  # shape = (n_el*n_el, D)
    mean_weights = arr.reshape(n_element, n_element, D)

    return alpha_grid, beta_grid, mean_weights


def main():
    # 配置参数
    n_element = 10      # 每维10格，共100个 (alpha,beta) 条件
    n_rounds  = 1_000_0  # 单次仿真轮数
    n_reps    = 1000       # 每条件下重复10次并平均

    # 并行网格实验
    alpha_grid, beta_grid, mean_weights = grid_experiment_parallel(
        n_element, n_rounds, n_reps
    )

    # 保存结果：将 (n,n,D) 展平成 (n*n, D) 写 CSV
    flat = mean_weights.reshape(-1, mean_weights.shape[-1])
    header = ",".join([f"w{k}" for k in range(flat.shape[1])])
    np.savetxt("multifeat_signal.csv",
               flat, delimiter=",",
               header=header, comments="")

    print("Done. Data saved to multifeat_signal.csv")

if __name__ == "__main__":
    main()
