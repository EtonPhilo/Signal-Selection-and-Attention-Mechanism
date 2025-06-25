import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math 

def attention_manipulate():
    df = pd.read_csv("attention_manipulation.csv")
    # 画图
    plt.figure(figsize=(8, 5))
    plt.plot(df.mean_attn2, color='tab:blue',  linestyle='solid', label="Biased Attention: P(attend F0)")
    plt.plot(df.mean_even2, color='tab:blue',  linestyle='dotted',  label="Biased Attention: P(even signals)")
    plt.plot(df.mean_attn1, color='tab:orange',  linestyle='solid', label="Biased Rewards: P(attend F0)")
    plt.plot(df.mean_even1, color='tab:orange',  linestyle='dotted',  label="Biased Rewards: P(even signals)")
    plt.xlabel("Rounds")
    plt.xlim([10,100000])
    plt.xscale("log")
    plt.ylabel("Probability")
    plt.title("Probability of signals associated with F0 and attention share for F0")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def game222_feat():
    pf = pd.read_csv("222game_feature0.csv")
    x = np.arange(1000,1000_001,1000)
    plt.figure(figsize=(10,6))
    plt.plot(x,pf.feat1, label="Number of Features = 1")
    plt.plot(x,pf.feat2, label="Number of Features = 2")
    plt.plot(x,pf.feat3, label="Number of Features = 3")
    plt.plot(x,pf.feat4, label="Number of Features = 4")
    plt.plot(x,pf.feat5, label="Number of Features = 5")
    plt.xscale("log")
    plt.xlim([1000,1000_000])
    plt.xlabel("Rounds")
    plt.ylabel("Attention share for F0")
    plt.title("Average attention to F0 under different feature counts")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

def xlogx(x,y):
    return x * np.log(x / y) 

def mutual_info(diff, agrid,bgrid,ps0,ps1,ps2):
    ones = np.ones(len(ps0), dtype=float)
    agrid1 = np.minimum(agrid - (ones * diff), ones * 0.999999) # 避免出现零点导致错误
    pf00 = ps0 + (ones - agrid) * (ones - bgrid) * ps1 + (ones - agrid1) * (ps2 /2)
    pf10 = agrid * (ones - bgrid) * ps1 + agrid1  * (ps2 /2)
    pf11 = agrid * bgrid * ps1 + agrid1 * (ps2 /2)
    pf01 = (ones - agrid) * bgrid * ps1 + (ones - agrid1) * (ps2 /2)
    info_s0 = - ps0 * np.log(pf00)
    info_s1 = xlogx(agrid * bgrid, pf11) + xlogx((ones - agrid) * bgrid, pf01) + xlogx(agrid * (ones - bgrid), pf10) + xlogx((ones - agrid) * (ones - bgrid), pf00)
    info_s1 *= ps1
    info_s2 = xlogx(agrid1 /2, pf11) + xlogx((ones - agrid1) /2, pf01) + xlogx(agrid1 /2, pf10) + xlogx((ones - agrid1) /2, pf00)
    info_s2 *= ps2
    return info_s0, info_s1, info_s2

def multifeature():
    # 设定格点数量以及信号s1和信号s2触发f0的正确率的差值
    n_element = 20
    diff = 0
    # 1. 读入 DataFrame
    df = pd.read_csv("multifeat_2020_diff=0.csv")

    # 2. 取出所有列变成 (n*n, D) 数组
    flat = df.values    # shape = (n*n, 16)

    # 3. reshape 回原始网格
    mean_weights = flat.reshape(n_element, n_element, 16)

    step = 0.5 / n_element
    mids = 0.5 + (np.arange(n_element) + 0.5) * step
    alpha_grid, beta_grid = np.meshgrid(mids, mids, indexing='ij')

    #   mean_weights[..., 0:3]   = state0 发 3 个信号的权重
    #   mean_weights[..., 3:6]   = state1 发 3 个信号的权重

    # 1. 抽出两种状态下的信号权重
    w0 = mean_weights[..., 0:3]   # (n, n, 3)
    w1 = mean_weights[..., 3:6]   # (n, n, 3)

    # 2. 归一化得到每个信号的发送概率 p(signal|state)
    p0 = w0 / w0.sum(axis=2, keepdims=True)  # (n, n, 3)
    p1 = w1 / w1.sum(axis=2, keepdims=True)  # (n, n, 3)

    # 3. 计算平均使用率：假设两种 state 等概率出现
    #    p_s1 = 0.5*[p0[...,1] + p1[...,1]]
    #    p_s2 = 0.5*[p0[...,2] + p1[...,2]]
    p_s0 = 0.5 * (p0[..., 0] + p1[..., 0])
    p_s1 = 0.5 * (p0[..., 1] + p1[..., 1])
    p_s2 = 0.5 * (p0[..., 2] + p1[..., 2])

    # 计算信息量差异
    info_s0, info_s1, info_s2 = mutual_info(diff, alpha_grid, beta_grid, p_s0, p_s1, p_s2)
    fgrid =  (info_s1 - info_s2 ) 

    # 4. 差值矩阵 Δ = p_s1 - p_s2
    delta = p_s1 - p_s2   # 形状 (n, n)

    # 5. 绘制示意图
    fig, ax = plt.subplots(figsize=(12, 6))

    #######################
    # 左侧子图：热力图
    #######################
    im = ax.imshow(
        delta.T,
        origin='lower',
        extent=[alpha_grid.min(), alpha_grid.max(),beta_grid.min(), beta_grid.max()],
        aspect='auto',
        cmap='viridis',
        vmin= np.min(delta),
        vmax= np.max(np.abs(delta))
    )
    # 在当前轴上添加 colorbar
    cb = fig.colorbar(im, ax = ax, label='Usage difference')
    cb.set_label('Usage difference')
    

    # 调整等高线：增大线宽、设定线型、颜色与透明度
    levels = np.linspace(fgrid.min(), fgrid.max(), 7)  # 5个层级，根据需要调整
    contour_lines = ax.contour(
        alpha_grid,  # X 轴坐标（extent 与 imshow 对应）
        beta_grid,# Y 轴坐标
        fgrid,
        levels=levels,
        colors='white',       # 线条颜色
        linewidths=2,         # 线条宽度，根据需要加粗
        linestyles='solid',   # 线条样式
        alpha= 0.8           # 调整透明度（0-1）
    )
    ax.clabel(contour_lines, inline=True, fontsize=10, fmt="%.2f")
    ax.set_label('Information quantity difference of signal 1 and signal 2')

    ax.set_xlabel('a (accuracy of feature 0)')
    ax.set_ylabel('b (accuracy of feature 1)')
    ax.set_title('Usage difference and informaiton quantity difference')

    plt.tight_layout()
    plt.show()

#game222_feat()

#attention_manipulate() 

multifeature()
