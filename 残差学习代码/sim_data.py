"""第一步：造数据。

跑完生成 data/samples.csv，每条样本包含某一时刻的状态、控制量，以及下一时刻的真实状态。
"""
import numpy as np
import pandas as pd
from params import P


def deriv(x, u, p, truth=False, d_eff=None):
    """状态导数。truth=True 用非线性轮胎+参数扰动，否则用线性轮胎（标称模型）。

    状态 x = [X, Y, psi, vx, vy, r]
    输入 u = [delta, ax]
    """
    X, Y, psi, vx, vy, r = x
    d, ax = u
    vs = max(abs(vx), 1.0)

    if truth:
        de = d_eff
        Cf_now = p['Cf'] * (1 + 0.2 * np.sin(0.5 * vx))   # 侧偏刚度随载荷变化
        af = de - (vy + p['lf'] * r) / vs
        ar = (p['lr'] * r - vy) / vs
        Ff = p['Df'] * np.sin(p['Cc'] * np.arctan(p['Bf'] * af))
        Fr = p['Dr'] * np.sin(p['Cc'] * np.arctan(p['Br'] * ar))
        Ff = Ff * (Cf_now / p['Cf'])
    else:
        de = d
        af = d - (vy + p['lf'] * r) / vs
        ar = (p['lr'] * r - vy) / vs
        Ff = p['Cf'] * af
        Fr = p['Cr'] * ar

    dvx = ax + vy * r
    dvy = (Ff * np.cos(de) + Fr) / p['m'] - vx * r
    dr = (p['lf'] * Ff * np.cos(de) - p['lr'] * Fr) / p['Iz']
    dX = vx * np.cos(psi) - vy * np.sin(psi)
    dY = vx * np.sin(psi) + vy * np.cos(psi)
    return np.array([dX, dY, r, dvx, dvy, dr])


def rk4(x, u, p, truth, d_eff, dt):
    k1 = deriv(x, u, p, truth, d_eff)
    k2 = deriv(x + 0.5 * dt * k1, u, p, truth, d_eff)
    k3 = deriv(x + 0.5 * dt * k2, u, p, truth, d_eff)
    k4 = deriv(x + dt * k3, u, p, truth, d_eff)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def build_ref(kind, ds=1.0):
    """三条参考线，按曲率序列积分出坐标。曲率×长度决定转角，务必让转角合理。"""
    if kind == 'A':                      # 直道 + 90 度大弯 + 180 度大弯
        seg = [(200, 0.0), (62.8, 1.0 / 40), (200, 0.0), (125.7, 1.0 / 40), (200, 0.0)]
    elif kind == 'B':                    # S 弯，每半个周期航向摆动约 45 度
        seg = [(880, None)]
    else:                                # 急弯：两个 90 度弯，半径 20 m
        seg = [(250, 0.0), (31.4, 1.0 / 20), (150, 0.0), (31.4, -1.0 / 20), (200, 0.0)]

    if kind == 'B':
        L = 880.0
        s = np.arange(0, L, ds)
        kap = 0.020 * np.sin(2 * np.pi * s / 120.0)
    else:
        L = sum(x[0] for x in seg)
        s = np.arange(0, L, ds)
        kap = np.zeros_like(s)
        acc = 0.0
        for ln, k in seg:
            kap[(s >= acc) & (s < acc + ln)] = k
            acc += ln

    w = max(3, int(round(15 / ds)))                 # 曲率做 15 m 平滑，模拟缓和曲线
    kap = np.convolve(kap, np.ones(w) / w, mode='same')
    psi = np.concatenate([[0.0], np.cumsum(kap[:-1] * ds)])
    X = np.concatenate([[0.0], np.cumsum(np.cos(psi[:-1]) * ds)])
    Y = np.concatenate([[0.0], np.cumsum(np.sin(psi[:-1]) * ds)])
    vref = np.minimum(P['vmax'],
                      np.sqrt(P['speed_factor'] * P['mu'] * P['g'] / np.maximum(np.abs(kap), 1e-3)))
    return dict(s=s, X=X, Y=Y, psi=psi, kap=kap, vref=vref, L=L)


def nearest(ref, x):
    d2 = (ref['X'] - x[0]) ** 2 + (ref['Y'] - x[1]) ** 2
    i = int(np.argmin(d2))
    return i, ref['psi'][i], ref['X'][i], ref['Y'][i]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def lateral_error(ref, x):
    """横向误差，正值表示车在参考线右侧（需要往左打方向）。"""
    _, pr, xr, yr = nearest(ref, x)
    return (x[0] - xr) * np.sin(pr) - (x[1] - yr) * np.cos(pr)


def stanley(x, ref):
    i, pr, _, _ = nearest(ref, x)
    e = lateral_error(ref, x)
    ff = (P['lf'] + P['lr']) * ref['kap'][i]          # 曲率前馈
    de = ff + wrap(pr - x[2]) + np.arctan2(1.2 * e, abs(x[3]) + 0.5)
    de = float(np.clip(de, -0.35, 0.35))
    ax = float(np.clip(0.6 * (ref['vref'][i] - x[3]), -6.0, 3.0))
    return np.array([de, ax])


def run_track(kind, laps=1):
    ref = build_ref(kind)
    p, dt = P, P['dt']
    x = np.array([ref['X'][0], ref['Y'][0], ref['psi'][0], 10.0, 0.0, 0.0])
    hist = [0.0] * max(p['delay'], 1)
    n_step = int(laps * ref['L'] / 11.0 / dt)
    rows = []
    for k in range(n_step):
        u = stanley(x, ref)
        hist.append(u[0])
        d_eff = hist[-1 - p['delay']]
        xn = rk4(x, u, p, True, d_eff, dt)
        rows.append([k, kind, *x, u[0], u[1], *xn])
        x = xn
    cols = ['step', 'track', 'X', 'Y', 'psi', 'vx', 'vy', 'r',
            'delta', 'ax', 'X2', 'Y2', 'psi2', 'vx2', 'vy2', 'r2']
    return pd.DataFrame(rows, columns=cols)


if __name__ == '__main__':
    all_df = []
    for kind in ['A', 'B', 'C']:
        df = run_track(kind, laps=1)
        print(f'轨道 {kind}: {len(df)} 条样本')
        all_df.append(df)
    data = pd.concat(all_df, ignore_index=True)
    data.to_csv('data/samples.csv', index=False)
    print(f'共 {len(data)} 条，已保存 data/samples.csv')
