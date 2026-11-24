"""预测与积分工具，训练脚本和控制脚本共用。"""
import numpy as np
import joblib
from params import P
from sim_data import deriv


def load_residual(path='models/residual.pkl'):
    """把 sklearn 的 MLP 权重取出来，改用手写前向，速度快很多。"""
    m = joblib.load(path)
    net, sc = m['net'], m['scaler']
    ts = m['tscaler']
    M = dict(W=[w.copy() for w in net.coefs_],
             b=[b.copy() for b in net.intercepts_],
             mean=sc.mean_.copy(), std=np.where(sc.scale_ > 1e-8, sc.scale_, 1.0),
             tmean=ts.mean_.copy(), tstd=ts.scale_.copy())
    return M


def nn_forward(M, z):
    h = (z - M['mean']) / M['std']
    for i, (w, b) in enumerate(zip(M['W'], M['b'])):
        h = h @ w + b
        if i < len(M['W']) - 1:
            h = np.tanh(h)
    return h * M['tstd'] + M['tmean']           # 反标准化回物理量


def feat(x, u):
    return np.array([x[3], x[4], x[5], u[0], u[1]])


def deriv_hyb(x, u, M, p=P):
    d = deriv(x, u, p, False)
    d[3:6] += np.clip(nn_forward(M, feat(x, u)), [-2.0, -6.0, -4.0], [2.0, 6.0, 4.0])
    return d


def deriv_kin(x, u, p=P):
    vx, d = x[3], u[0]
    L = p['lf'] + p['lr']
    return np.array([vx * np.cos(x[2]), vx * np.sin(x[2]),
                     vx * np.tan(np.clip(d, -0.35, 0.35)) / L,
                     u[1], 0.0, 0.0])


def integ(x, u, dt, dfun):
    k1 = dfun(x, u)
    k2 = dfun(x + 0.5 * dt * k1, u)
    k3 = dfun(x + 0.5 * dt * k2, u)
    k4 = dfun(x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def plant(x, u, dt, d_eff):
    """真实车辆（被控对象）：非线性轮胎，控制器并不知道这些。"""
    return integ(x, u, dt, lambda a, b: deriv(a, b, P, True, d_eff))


def predict(x, u, dt, mode='nom', M=None):
    """mode: 'nom' 标称 / 'hyb' 标称+残差 / 'kin' 纯运动学。"""
    if mode == 'nom':
        return integ(x, u, dt, lambda a, b: deriv(a, b, P, False))
    if mode == 'kin':
        return integ(x, u, dt, lambda a, b: deriv_kin(a, b, P))
    return integ(x, u, dt, lambda a, b: deriv_hyb(a, b, M))
