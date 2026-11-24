"""第二步：训练残差网络，并统计开环预测误差。

输出：models/residual.pkl、results/net_loss.png、results/residual_scatter.png、results/openloop.csv
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from params import P
from sim_data import deriv
from model_lib import predict

df = pd.read_csv('data/samples.csv')
dt = P['dt']

# ---------- 1. 构造特征与标签 ----------
feat, targ, track, ay = [], [], [], []
for row in df.itertuples():
    x = np.array([row.X, row.Y, row.psi, row.vx, row.vy, row.r])
    u = np.array([row.delta, row.ax])
    f_nom = deriv(x, u, P, False)[3:6]                       # 标称模型给出的导数
    true_rate = (np.array([row.vx2, row.vy2, row.r2]) - x[3:6]) / dt
    feat.append([row.vx, row.vy, row.r, row.delta, row.ax])
    targ.append(true_rate - f_nom)                            # 残差标签
    track.append(row.track)
    ay.append(abs(row.vx * row.r))

F = np.array(feat)
T = np.array(targ)
# 划分：每条线前 60% 用于训练，后 40% 用于测试
track = np.array(track)
frac = np.zeros(len(track))
for k in ['A', 'B', 'C']:
    idx = np.where(track == k)[0]
    frac[idx] = np.arange(len(idx)) / len(idx)
tr = frac < 0.6
te = frac >= 0.6

print('残差幅值（训练集）均值/最大：', np.round(np.abs(T[tr]).mean(0), 3),
      np.round(np.abs(T[tr]).max(0), 3))

# ---------- 2. 训练 ----------
sc = StandardScaler().fit(F[tr])
keep = tr & (np.abs(T[:, 1]) < 4.0) & (np.abs(T[:, 2]) < 3.0)     # 去掉极端离群样本
ts = StandardScaler().fit(T[keep])
net = MLPRegressor(hidden_layer_sizes=(128, 128), activation='tanh',
                   solver='adam', learning_rate_init=1e-3, max_iter=600,
                   early_stopping=True, n_iter_no_change=40, random_state=0)
net.fit(sc.transform(F[keep]), ts.transform(T[keep]))
joblib.dump({'net': net, 'scaler': sc, 'tscaler': ts}, 'models/residual.pkl')
from model_lib import load_residual
M = load_residual()

pred_te = ts.inverse_transform(net.predict(sc.transform(F[te])))
rmse = np.sqrt(((pred_te - T[te]) ** 2).mean(0))
print('测试集各分量残差 RMSE：', np.round(rmse, 4), ' 迭代轮数：', net.n_iter_)

plt.figure(figsize=(5, 3.2))
plt.plot(net.loss_curve_, label='train loss')
plt.xlabel('iteration'); plt.ylabel('MSE'); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig('results/net_loss.png', dpi=200); plt.close()

plt.figure(figsize=(5, 3.2))
plt.scatter(np.array(ay)[te], np.linalg.norm(T[te], axis=1), s=3, alpha=.4, label='真实残差')
plt.scatter(np.array(ay)[te], np.linalg.norm(pred_te, axis=1), s=3, alpha=.4, label='网络预测')
plt.xlabel('|a_y| (m/s^2)'); plt.ylabel('残差范数'); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig('results/residual_scatter.png', dpi=200); plt.close()

# ---------- 3. 开环多步预测误差 ----------
d = df.reset_index(drop=True)
starts = [i for i in range(0, len(d) - 12, 30) if track[i] == track[i + 11]]


def roll(i, k, mode):
    x = np.array([d.X[i], d.Y[i], d.psi[i], d.vx[i], d.vy[i], d.r[i]])
    for j in range(k):
        u = np.array([d.delta[i + j], d.ax[i + j]])
        x = predict(x, u, dt, mode, M)
    truth = np.array([d.vx[i + k], d.vy[i + k], d.r[i + k]])
    return x[3:6] - truth


rows = []
for k in [1, 5, 10]:
    for mode, name in [('nom', '标称模型'), ('hyb', '标称+残差')]:
        err = np.array([roll(i, k, mode) for i in starts])
        rows.append([k, name, *np.sqrt((err ** 2).mean(0)), np.sqrt((err ** 2).sum(1).mean())])
tab = pd.DataFrame(rows, columns=['步数', '模型', 'vx RMSE', 'vy RMSE', 'r RMSE', '综合'])
tab.to_csv('results/openloop.csv', index=False)
print(tab.round(3).to_string(index=False))
