"""pytest 共享装置。

src/ 布局在未执行 ``pip install -e .`` 时也需可测试，故显式把 ``src`` 加入 sys.path。
"""

from __future__ import annotations

import math
import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from hover_rotor.config import RotorInputs  # noqa: E402
from hover_rotor.geometry.cst import fit_shape_weights  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _force_matplotlib_agg() -> None:
    """CI 与本地无显示环境下绘图不得弹出后端窗口。"""
    import matplotlib

    matplotlib.use("Agg", force=True)


@pytest.fixture(scope="session")
def baseline_inputs() -> RotorInputs:
    """计划.md Task 3.2 的基准验收工况：T=80 N、R=0.75 m、t_te=2.5 mm。"""
    return RotorInputs(thrust=80.0, radius=0.75, te_thickness=0.0025)


def naca4_surface(geom: str, n_points: int = 1200):
    """4 位 NACA 系列上下表面点阵 (x, y)，x 单调递增。

    仅测试夹具使用：给极线代理一个**已知解析厚度/弯度分布**的基准剖面，
    从而能用地磁文献值（NACA 2412 的 aL0=-2.1 deg、c_m,c4=-0.05）钉死薄翼实现。
    """
    m, p, t = int(geom[0]) / 100.0, int(geom[1]) / 10.0, int(geom[2:]) / 100.0
    xi = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, n_points)))
    yt = 5.0 * t * (0.2969 * np.sqrt(xi) - 0.1260 * xi - 0.3516 * xi**2 + 0.2843 * xi**3 - 0.101516 * xi**4)
    if m == 0.0 or p == 0.0:
        yc = np.zeros_like(xi)
    else:
        yc = np.where(
            xi < p,
            m / p**2 * (2.0 * p * xi - xi**2),
            m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * xi - xi**2),
        )
    theta = np.arctan(np.gradient(yc, xi))
    return xi - yt * np.sin(theta), yc + yt * np.cos(theta), xi + yt * np.sin(theta), yc - yt * np.cos(theta)


def cst_from_naca(geom: str, n: int = 6, dz_te: float = 0.001):
    """把 4 位 NACA 剖面嵌入带钝尾缘楔形的 CST 权重空间。

    楔形抵消约定：上表面生成式含 ``+psi*dz/2``，故拟合时扣 ``+dz_te``；下表面生成式含
    ``-psi*dz/2``，故拟合时扣 ``-dz_te``。两次扣除后 C*S 分别还原 yu 与 yl，
    重新生成即得「中弧线不变、厚度在尾缘对称张开 dz_te」的平尾缘剖面。
    符号写反会让 0.001 的楔形泄漏进中弧线，把对称剖面的零升迎角污染成 ~0.4 deg。
    """
    psi = np.linspace(1.0e-5, 1.0, 800)
    xu, yu, xl, yl = naca4_surface(geom)
    au = fit_shape_weights(psi, np.interp(psi, xu, yu), n, dz_te=dz_te)
    al = fit_shape_weights(psi, np.interp(psi, xl, yl), n, dz_te=-dz_te)
    return au, al


@pytest.fixture(scope="session")
def naca_to_cst():
    """返回「4 位 NACA 剖面 -> 钝尾缘 CST 权重」工厂函数，供各测试构造基准翼型。"""
    return cst_from_naca


@pytest.fixture(scope="session")
def naca0012_cst():
    """Caradonna-Tung 试验与 §2.1 验收所用的 NACA 0012 等效 CST 权重（微钝尾缘）。"""
    return cst_from_naca("0012")
