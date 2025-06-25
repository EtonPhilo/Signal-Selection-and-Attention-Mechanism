import random
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

def simulate_attention(n_features: int,
                       num_rounds: int,
                       seed: int) -> list:
    """
    并行单次模拟：只返回在各采样点的 特征 0 注意力占比历史
    """
    rng = random.Random(seed)

    # Sender urns for state 0/1  → weights for signal 0/1
    sender_urn = {0: [1.0, 1.0], 1: [1.0, 1.0]}
    # Attention urn: n_features 个特征，各初始权重 1.0
    attention_urn = [1.0] * n_features
    # Receiver action urns: for each feature i, each observed value v∈{0,1}, 两个动作初始权重 1.0
    receiver_action_urn = [
        {0: [1.0, 1.0], 1: [1.0, 1.0]}
        for _ in range(n_features)
    ]

    feat0_history = []

    for t in range(1, num_rounds + 1):

        # 1. 随机抽 state
        state = rng.choice([0, 1])

        # 2. Sender 抽 signal
        sig_weights = sender_urn[state]
        signal = rng.choices([0, 1], weights=sig_weights, k=1)[0]

        # 3. 构造特征向量：特征 0 = signal，其余随机
        features = [None]*n_features
        features[0] = signal
        for i in range(1, n_features):
            features[i] = rng.choice([0, 1])

        # 4. Receiver 抽 attention 特征 f_choice
        f_choice = rng.choices(range(n_features), weights=attention_urn, k=1)[0]
        obs = features[f_choice]

        # 5. Receiver 抽 action
        act_weights = receiver_action_urn[f_choice][obs]
        action = rng.choices([0,1], weights=act_weights, k=1)[0]

        # 6. 强化更新（仍然只对“成功”（action==state）才更新 urns）
        if action == state:
            sender_urn[state][signal] += 1.0
            attention_urn[f_choice]  += 1.0
            receiver_action_urn[f_choice][obs][action] += 1.0

        # 每1000轮记录一次特征 0 注意力占比
        if t % 1000 == 0:
            total_att = sum(attention_urn)
            feat0_history.append(attention_urn[0] / total_att)

    return feat0_history


def run_simulations(n_features: int,
                    num_simulations: int,
                    num_rounds: int):
    """
    并行地跑 num_simulations 次 simulate_attention，
    注意力曲线
    """

    # 为每次模拟准备 seed
    seeds = list(range(num_simulations))
    args = [(n_features, num_rounds, s) for s in seeds]

    # 并行执行
    n_workers = max(1, cpu_count() - 1)
    sum_hist = np.zeros(num_rounds // 1000, dtype=np.float64)    
    with Pool(n_workers) as pool:
        for history in pool.starmap(simulate_attention, args):
            sum_hist += history

    # 将总数除以模拟次数得到平均值
    avg_feat0 = sum_hist / num_simulations

    return avg_feat0


def main():
    conditions      = [1,2,3,4,5]   # 不同特征数量
    num_simulations = 1000
    num_rounds      = 1_000_000

    # 存储每种条件下的平均曲线
    all_results = np.zeros((5,num_rounds // 1000),dtype=float)

    for n in conditions:
        avg_history = run_simulations(n, num_simulations, num_rounds)
        all_results[n-1,:] = avg_history

    data=all_results.T
    # 把结果写入 CSV
    header = "feat1,feat2,feat3,feat4,feat5"
    np.savetxt("222game_feature0.csv", data, delimiter=",", header=header, comments="")

if __name__ == "__main__":
    main()
    print("Program finished!\n The data is collected in 222game_feature0.csv!")
