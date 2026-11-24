"""第三步：闭环对比。

三组控制器（标称 / 标称+残差 / 纯运动学）跑三条参考线。
控制方式：每 0.25 s 解一次，预测 4 步，优化"未来一段保持同一转角"的那个转角。
输出 results/metrics.csv、results/traj.png、results/err.png
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from scipy.optimize import minimize_scalar

from params import P
from sim_data import build_ref, nearest, lateral_error
from model_lib import predict, plant, load_residual

N = 4                 # 预测步数
DT_MPC = 0.25         # 控制周期 0.25 s（约 4 Hz，接近实车控制器频率）


def speed_cmd(x, ref, i):
    return float(np.clip(0.6 * (ref['vref'][i] - x[3]), -6.0, 3.0))


def solve(x, ref, i, mode, M):
    """单变量：在 [-0.35, 0.35] 里找未来 N 步横向误差最小的那个转角。"""
    ax = speed_cmd(x, ref, i)

    def cost(d):
        xs = x.copy()
        J = 0.0
        for k in range(N):
            xs = predict(xs, np.array([d, ax]), DT_MPC, mode, M)
            J += lateral_error(ref, xs) ** 2
        return J

    r = minimize_scalar(cost, bounds=(-0.35, 0.35), method='bounded',
                        options={'xatol': 1e-3})
    return float(r.x), bool(r.success)


def run(kind, mode, M=None, max_step=4000):
    ref = build_ref(kind)
    x = np.array([ref['X'][0], ref['Y'][0], ref['psi'][0], 10.0, 0.0, 0.0])
    log, n_fail, t_solve = [], 0, []
    for k in range(max_step):
        i, *_ = nearest(ref, x)
        t0 = time.perf_counter()
        d, ok = solve(x, ref, i, mode, M)
        t_solve.append(time.perf_counter() - t0)
        n_fail += (not ok)
        u = np.array([d, speed_cmd(x, ref, i)])
        x = plant(x, u, DT_MPC, u[0])
        log.append([k * DT_MPC, x[0], x[1], x[3], lateral_error(ref, x)])
        if i >= len(ref['s']) - 2:
            break
    df = pd.DataFrame(log, columns=['t', 'X', 'Y', 'v', 'ey'])
    m = dict(track=kind, mode=mode,
             ey_rms=float(np.sqrt((df.ey ** 2).mean())),
             ey_max=float(df.ey.abs().max()),
             v_mean=float(df.v.mean()),
             t_lap=float(df.t.iloc[-1]),
             solve_ms=float(np.mean(t_solve) * 1000),
             fail_rate=float(n_fail / len(t_solve)))
    return df, m


if __name__ == '__main__':
    M = load_residual()
    rows, trajs = [], {}
    for kind in ['A', 'B', 'C']:
        for mode, name in [('nom', '标称模型'), ('hyb', '标称+残差'), ('kin', '纯运动学')]:
            df, m = run(kind, mode, M)
            trajs[(kind, name)] = df
            rows.append([kind, name, m['ey_rms'], m['ey_max'], m['v_mean'],
                         m['t_lap'], m['solve_ms'], m['fail_rate']])
            print(f'{kind} {name}: ey_rms={m["ey_rms"]:.3f} 单步 {m["solve_ms"]:.1f} ms')
    tab = pd.DataFrame(rows, columns=['轨道', '控制器', '横向误差RMS', '横向误差最大',
                                      '平均车速', '单圈时间s', '单步耗时ms', '求解失败率'])
    tab.to_csv('results/metrics.csv', index=False)
    print(tab.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, kind in zip(axes, ['A', 'B', 'C']):
        ref = build_ref(kind)
        ax.plot(ref['X'], ref['Y'], 'k--', lw=1, label='参考线')
        for name, c in [('标称模型', 'tab:blue'), ('标称+残差', 'tab:red'), ('纯运动学', 'tab:green')]:
            t = trajs[(kind, name)]
            ax.plot(t.X, t.Y, color=c, lw=1.2, label=name)
        ax.set_aspect('equal'); ax.set_title(f'轨道 {kind}'); ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig('results/traj.png', dpi=200); plt.close()

    plt.figure(figsize=(7, 3))
    for name, c in [('标称模型', 'tab:blue'), ('标称+残差', 'tab:red'), ('纯运动学', 'tab:green')]:
        t = trajs[('C', name)]
        plt.plot(t.t, t.ey, color=c, label=name)
    plt.xlabel('时间 (s)'); plt.ylabel('横向误差 (m)'); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig('results/err.png', dpi=200); plt.close()
