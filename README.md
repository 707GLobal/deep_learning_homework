# 基于残差学习的车辆动力学建模与轨迹跟踪

用简化的单轨模型做轨迹跟踪，训练一个小神经网络学习该模型与高保真模型之间的状态导数残差，对比三组控制器（标称模型 / 标称+残差 / 纯运动学）的闭环跟踪精度。纯离线仿真，CPU 即可运行。

## 目录结构

```
.
├── README.md                 本文件，架构与用法
└── 残差学习代码/
    ├── params.py             车辆与仿真参数
    ├── sim_data.py           仿真与数据采集
    ├── model_lib.py          状态导数、积分、预测（三种模型）
    ├── train_residual.py     残差网络训练与开环误差统计
    ├── mpc_eval.py           三组控制器闭环对比
    ├── requirements.txt      依赖清单
    ├── data/                 运行产物：samples.csv
    ├── models/               运行产物：residual.pkl
    └── results/              运行产物：图与指标表
```

## 环境

| 项目 | 要求 |
|---|---|
| Python | 3.10 – 3.13（推荐 3.11） |
| numpy / scipy | ≥ 1.24 / ≥ 1.10 |
| pandas / scikit-learn | ≥ 2.0 / ≥ 1.3 |
| matplotlib / joblib | ≥ 3.7 / ≥ 1.3 |
| 硬件 | 仅 CPU，内存 8 GB 足够 |

## 运行

```powershell
conda create -n dlhw python=3.11 -y
conda activate dlhw
cd "C:\Users\User\Desktop\深度学习内容\大作业\残差学习代码"
python -m pip install -r requirements.txt

python sim_data.py          # 造数据，约 2 秒
python train_residual.py    # 训练残差网络，1–2 分钟
python mpc_eval.py          # 闭环对比，约 1 分钟
```

三步必须按顺序执行，后一步依赖前一步的产物。

## 产物

- `data/samples.csv`：约 1 万条状态-控制-次态样本
- `models/residual.pkl`：残差网络权重与标准化参数
- `results/`：`net_loss.png`、`residual_scatter.png`、`traj.png`、`err.png`、`metrics.csv`、`openloop.csv`

三个目录初始为空，首次运行后生成。
