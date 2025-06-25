import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

def run_metrics(n_iters: int,
                init_attention,
                diff_reward: bool,
                seed: int):
    """
    单次模拟，返回 even_prop, attendF0_prop 两个 ndarray
    even_prop[t]    = P(偶数信号) at iteration t
    attendF0_prop[t] = P(关注 F0) at iteration t
    """
    rng = np.random.default_rng(seed)
    # sender urn: 2 states × 4 signals
    sender_urn    = np.ones((2, 4), dtype=float)
    # receiver action urns: f∈{0,1} × v∈{0,1} × a∈{0,1}
    action_urn    = np.ones((2, 2, 2), dtype=float)
    # 注意力 URN：F0 vs F1
    attention_urn = np.array(init_attention, dtype=float).copy()

    even_prop     = np.empty(n_iters, dtype=float)
    attendF0_prop = np.empty(n_iters, dtype=float)

    for t in range(n_iters):
        # 理论上的偶数信号概率 P(s∈{0,2}|state=0,1) 平均
        even_prop[t] = sender_urn[:, [0, 2]].sum() / sender_urn.sum()

        # 记录关注 F0 的概率
        attendF0_prop[t] = attention_urn[0] / attention_urn.sum()

        # —— 游戏流程 —— #
        state = rng.integers(0, 2)
        p_s = sender_urn[state] / sender_urn[state].sum()
        s   = rng.choice(4, p=p_s)

        # 信号到信号特征的固定映射 fv为对应的信号特征的值
        if s % 2 == 0:
            feature_class = 0
            fv = s // 2        # s=0→fv=0, s=2→fv=1
        else:
            feature_class = 1
            fv = (s - 1) // 2  # s=1→fv=0, s=3→fv=1

        # 接收者选注意力特征
        p_alpha  = attention_urn / attention_urn.sum()
        f_choice = rng.choice(2, p=p_alpha)
        fn = rng.integers(0,2) #噪声特征

        # 动作选择
        if f_choice == feature_class:
            p_a = action_urn[f_choice, fv] / action_urn[f_choice, fv].sum() # 注意的信号恰好是信号特征
        else:
            p_a = action_urn[f_choice, fn] / action_urn[f_choice, fn].sum() # 注意的信号是噪声
        a = rng.choice(2, p=p_a)

        # 奖励
        success = (a == state)
        if not success:
            reward = 0.0
        else:
            reward = (2.0 if f_choice == 0 else 1.0) if diff_reward else 1.0

        # 强化学习
        attention_urn[f_choice] += reward #根据设置条件进行奖励
        action_urn[f_choice, fv, a] += min(reward,1.0) #如果成功，则奖励1
        sender_urn[state,s] += min(reward,1.0) #如果成功，则奖励1


    return even_prop, attendF0_prop


def worker_mean(args):
    """
    在一个进程里跑多次模拟，内部先累加求均，然后一次性返回两个平均曲线。
    args = (n_iters, init_attention, diff_reward, seed_start, runs_per_worker)
    """
    n_iters, init_att, diff_reward, seed0, runs = args
    # 两条曲线的累加器
    sum_even   = np.zeros(n_iters, dtype=float)
    sum_attn0  = np.zeros(n_iters, dtype=float)

    for i in range(runs):
        seed = seed0 + i
        even, attn0 = run_metrics(n_iters, init_att, diff_reward, seed)
        sum_even  += even
        sum_attn0 += attn0

    # 取平均后只返回一份
    return sum_even / runs, sum_attn0 / runs

def experiment_parallel(n_runs, n_iters, n_workers=None):
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    # 每个进程分担的 runs 数，最后可能有余数
    base = n_runs // n_workers
    rem  = n_runs % n_workers

    jobs = []
    # 情形1 jobs
    seed = 0
    for w in range(n_workers):
        runs = base + (1 if w < rem else 0)
        if runs == 0: break
        jobs.append((n_iters, [1,1], True,  seed, runs))
        seed += runs

    # 并行执行
    with Pool(n_workers) as pool:
        # out1 是 length=n_workers 的 [(mean_even,mean_attn0), ...]
        out1 = pool.map(worker_mean, jobs)

    # 把所有 worker 的平均再根据 runs 加权平均一次
    # 因为我们已经让每个 worker 内“平均”过了，所以这里各自权重是 runs，而总 weight = n_runs
    mean_even1 = sum(out1[i][0] * jobs[i][4] for i in range(len(out1))) / n_runs
    mean_attn1 = sum(out1[i][1] * jobs[i][4] for i in range(len(out1))) / n_runs

    # 同理，再跑情形2
    jobs2 = []
    seed = 10000
    for w in range(n_workers):
        runs = base + (1 if w < rem else 0)
        if runs == 0: break
        jobs2.append((n_iters, [10,1], False, seed, runs))
        seed += runs

    with Pool(n_workers) as pool:
        out2 = pool.map(worker_mean, jobs2)

    mean_even2 = sum(out2[i][0] * jobs2[i][4] for i in range(len(out2))) / n_runs
    mean_attn2 = sum(out2[i][1] * jobs2[i][4] for i in range(len(out2))) / n_runs

    # 1) 把它们合并成一个 n×4 的矩阵
    data = np.vstack([
    mean_even1,
    mean_attn1,
    mean_even2,
    mean_attn2
    ]).T   # 转置后每列就是一条曲线

    # 2) 保存到 CSV，第一行写列名
    header = "mean_even1,mean_attn1,mean_even2,mean_attn2"
    np.savetxt("attention_manipulation.csv", data, delimiter=",", header=header, comments="")


if __name__ == "__main__":
    # Windows 下一定要放在这里
    experiment_parallel(n_runs=1000, n_iters=100000, n_workers=8)
    print("Program Finished!\n The data is collected in attention_manipulation.csv!")