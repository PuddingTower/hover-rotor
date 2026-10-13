"""全局物理常数、默认设计边界与输入校验模型。

模块内所有物理量采用 SI 单位（m、N、s、kg）。仅制造文件导出层按加工惯例转换为 mm。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator

RHO_INF = 1.225
"""标准海平面空气密度 [kg/m^3]"""

A_INF = 340.3
"""标准海平面声速 [m/s]"""

MU_AIR = 1.81e-5
"""标准海平面空气动力粘度 [Pa*s]，雷诺数计算用。

取 1.81e-5（约 20 degC）而非 15 degC 的 1.789e-5：Caradonna-Tung 试验报告给出的
桨尖弦向雷诺数 1.92e6 只有在本值下才能由 rho*Omega*R*c/mu 复现，两者必须同源。
"""

DEFAULT_HUB_RATIO = 0.15
"""桨毂无量纲半径 r_hub/R，与十截面表首站位自洽"""

DEFAULT_BLADE_COUNT = 2
DEFAULT_TIP_MACH_MAX = 0.65
DEFAULT_N_SECTIONS = 10
DEFAULT_N_POINTS = 200

TE_CHORD_RATIO_LIMIT = 0.05
"""钝尾缘绝对厚度相对叶尖弦长的工艺上限"""


class OptionalDependencyError(RuntimeError):
    """可选后端（CAD 内核 / 神经网络权重）缺失。

    与 ImportError 区分：核心气动链路不应因导出层依赖缺席而不可用，
    调用方应捕获本异常并降级，而非中止整个流程。
    """


class FeasibilityError(ValueError):
    """设计边界与物理约束矛盾，优化不可行。"""


class RotorInputs(BaseModel):
    """端到端反设计的工程指标输入空间。

    仅描述用户给定的物理指标，不含任何几何自由变量——几何全部由优化器反演。
    """

    model_config = ConfigDict(extra="forbid")

    # 数值边界全部由下方 field_validator 承担。刻意不使用 pydantic 的 gt/ge/le 内建约束：
    # 内建约束会抢先命中并输出英文消息，令面向操作者的中文越界提示失效。
    thrust: float = Field(description="目标悬停推力 T [N]")
    radius: float = Field(description="旋翼外半径 R [m]")
    te_thickness: float = Field(description="钝尾缘绝对厚度 t_te [m]，全展恒定")

    blade_count: int = Field(default=DEFAULT_BLADE_COUNT)
    hub_ratio: float = Field(default=DEFAULT_HUB_RATIO)
    rho: float = Field(default=RHO_INF)
    a_inf: float = Field(default=A_INF)
    tip_mach_max: float = Field(default=DEFAULT_TIP_MACH_MAX)
    n_sections: int = Field(default=DEFAULT_N_SECTIONS)
    chord_tip: float | None = Field(
        default=None,
        description="预期叶尖弦长 [m]。给定时校验 t_te/c_tip 工艺上限；未给定时该约束由优化器边界间接保证",
    )

    @field_validator("thrust")
    @classmethod
    def _check_thrust(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"推力必须为正值，收到 {v:.4g} N")
        if not 1.0 <= v <= 20000.0:
            raise ValueError(f"推力 {v:.3g} N 超出可设计范围 [1.0, 20000.0] N")
        return v

    @field_validator("radius")
    @classmethod
    def _check_radius(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"旋翼半径必须为正值，收到 {v:.4g} m")
        if not 0.05 <= v <= 5.0:
            raise ValueError(f"旋翼半径 {v:.4g} m 超出可设计范围 [0.05, 5.0] m")
        return v

    @field_validator("te_thickness")
    @classmethod
    def _check_te_range(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(
                f"钝尾缘厚度必须为正值，收到 {v * 1000:.4g} mm；尖尾缘（t_te = 0）不被本系统接受"
            )
        if not 3.0e-4 <= v <= 5.0e-3:
            raise ValueError(
                f"钝尾缘厚度 {v * 1000:.3g} mm 超出工艺范围 [0.3, 5.0] mm；尖尾缘（t_te -> 0）不被本系统接受"
            )
        return v

    @field_validator("blade_count")
    @classmethod
    def _check_blade_count(cls, v: int) -> int:
        if not 2 <= v <= 6:
            raise ValueError(f"叶片数 {v} 超出可行范围 [2, 6]")
        return v

    @field_validator("hub_ratio")
    @classmethod
    def _check_hub_ratio(cls, v: float) -> float:
        if not 0.05 <= v <= 0.20:
            raise ValueError(f"轮毂比 r_hub/R = {v:.3g} 超出可行范围 [0.05, 0.20]")
        return v

    @field_validator("rho", "a_inf")
    @classmethod
    def _check_positive_medium(cls, v: float, info) -> float:
        if v <= 0.0:
            raise ValueError(f"{info.field_name} 必须为正值，收到 {v:.4g}")
        return v

    @field_validator("tip_mach_max")
    @classmethod
    def _check_tip_mach(cls, v: float) -> float:
        if not 0.0 < v <= 0.95:
            raise ValueError(f"叶尖马赫数上限 {v:.3g} 需落在 (0, 0.95] 内")
        return v

    @field_validator("n_sections")
    @classmethod
    def _check_n_sections(cls, v: int) -> int:
        if not 4 <= v <= 40:
            raise ValueError(f"展向截面数 {v} 需落在 [4, 40] 内")
        return v

    @field_validator("chord_tip")
    @classmethod
    def _check_te_feasibility(cls, v: float | None, info) -> float | None:
        if v is not None and v <= 0.0:
            raise ValueError(f"叶尖弦长必须为正值，收到 {v:.4g} m")
        t_te = info.data.get("te_thickness")
        if v is not None and t_te is not None and t_te / v > TE_CHORD_RATIO_LIMIT:
            raise ValueError(
                f"尾缘厚度比 t_te/c_tip = {t_te / v:.3%} 超过工艺上限 {TE_CHORD_RATIO_LIMIT:.0%}；"
                "请增大弦长或减小输入尾缘厚度"
            )
        return v

    @property
    def disk_area(self) -> float:
        """旋翼扫掠盘面积 A = pi R^2 [m^2]"""
        return math.pi * self.radius**2

    @property
    def omega_max(self) -> float:
        """叶尖马赫数约束下的最高转速 [rad/s]"""
        return self.tip_mach_max * self.a_inf / self.radius


# --------------------------------------------------------------------------------------
# 结果数据契约（Task 0.1 定义的"结果数据类"）
#
# 这是几何/气动/优化/CAD/可视化五层之间唯一的共享词汇。M4/M5 的导出器与绘图器
# 只读这里的字段，不得依赖优化器内部结构，因此本契约一旦定稿即对下游冻结。
# --------------------------------------------------------------------------------------


@dataclass
class SectionResult:
    """单个展向截面的完整档案：几何 + 气动工作点 + 截面结构力学量。

    坐标存在两份，刻意不互相换算：
      * ``points_normalized`` 是无量纲弦向坐标 (psi, zeta)，几何层与极线层的交换格式；
      * ``points_mm`` 是该截面放到真实弦长、真实半径后的**物理毫米**二维点阵，
        已绕桨轴旋转 ``theta_deg``，是 DXF/STEP 的直接输入。制造层不应再做任何缩放。
    """

    index: int
    r_over_R: float
    r: float
    chord: float
    theta_deg: float
    te_thickness: float
    dz_te: float
    alpha_deg: float
    cl: float
    cd: float
    cm: float
    re: float
    ma: float
    au: tuple[float, ...] = ()
    al: tuple[float, ...] = ()
    points_normalized: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 2)))
    points_mm: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 2)))
    dN: float = 0.0
    dQ: float = 0.0
    v_i: float = 0.0
    phi_deg: float = 0.0
    area: float = 0.0
    x_cg: float = 0.0
    y_cg: float = 0.0
    i_xx: float = 0.0
    j_torsion: float = 0.0

    @property
    def ld(self) -> float:
        return self.cl / self.cd if self.cd > 0.0 else 0.0

    @property
    def chord_over_R(self) -> float:
        return self.chord / self.r_over_R if self.r_over_R > 0.0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["points_normalized"] = self.points_normalized.tolist()
        payload["points_mm"] = self.points_mm.tolist()
        payload["ld"] = self.ld
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SectionResult:
        known = {f.name for f in fields(cls)}
        payload = {k: v for k, v in data.items() if k in known}
        for key in ("points_normalized", "points_mm"):
            payload[key] = np.asarray(payload.get(key, []), dtype=np.float64)
        for key in ("au", "al"):
            payload[key] = tuple(float(x) for x in payload.get(key, ()))
        return cls(**payload)  # type: ignore[arg-type]


@dataclass
class RotorResult:
    """一次反设计的终态：整机气动量 + 十截面档案 + 复现所需的配置与种子。

    ``to_dict``/``from_dict`` 必须无损往返（计划.md Task 4.3 判定准则：
    "JSON 可反序列化回等价 RotorResult"），故所有数组均以 list 形式落盘。
    """

    inputs: RotorInputs
    omega: float
    rpm: float
    tip_mach: float
    thrust: float
    power: float
    torque: float
    fm: float
    ct: float
    cp: float
    power_ideal: float
    sections: list[SectionResult] = field(default_factory=list)
    r_over_R: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    chord: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    theta_deg: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    v_i: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    circulation: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    cl_distribution: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    polars: list[dict[str, list[float]]] = field(default_factory=list)
    objective: float = 0.0
    iterations: int = 0
    convergence_history: list[dict[str, float]] = field(default_factory=list)
    seed: int | None = None
    backend: str = "analytic"
    solver_stage: str = "de+slsqp"
    elapsed_s: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def thrust_error(self) -> float:
        """终态推力相对目标推力的绝对误差（§5 红线要求 <= 0.5%）。"""
        return abs(self.thrust - self.inputs.thrust) / self.inputs.thrust

    @property
    def n_sections(self) -> int:
        return len(self.sections)

    def te_thickness_vector(self) -> NDArray[np.float64]:
        return np.array([s.te_thickness for s in self.sections], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "inputs": self.inputs.model_dump(),
            "omega": self.omega,
            "rpm": self.rpm,
            "tip_mach": self.tip_mach,
            "thrust": self.thrust,
            "power": self.power,
            "torque": self.torque,
            "fm": self.fm,
            "ct": self.ct,
            "cp": self.cp,
            "power_ideal": self.power_ideal,
            "sections": [s.to_dict() for s in self.sections],
            "r_over_R": self.r_over_R.tolist(),
            "chord": self.chord.tolist(),
            "theta_deg": self.theta_deg.tolist(),
            "v_i": self.v_i.tolist(),
            "circulation": self.circulation.tolist(),
            "cl_distribution": self.cl_distribution.tolist(),
            "polars": self.polars,
            "objective": self.objective,
            "iterations": self.iterations,
            "convergence_history": self.convergence_history,
            "seed": self.seed,
            "backend": self.backend,
            "solver_stage": self.solver_stage,
            "elapsed_s": self.elapsed_s,
            "warnings": self.warnings,
        }
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RotorResult:
        known = {f.name for f in fields(cls)}
        payload = {k: v for k, v in data.items() if k in known}
        payload["inputs"] = RotorInputs.model_validate(data["inputs"])
        payload["sections"] = [SectionResult.from_dict(s) for s in data.get("sections", [])]
        for key in ("r_over_R", "chord", "theta_deg", "v_i", "circulation", "cl_distribution"):
            payload[key] = np.asarray(payload.get(key, []), dtype=np.float64)
        return cls(**payload)  # type: ignore[arg-type]


RotorHoverResult = RotorResult
"""计划.md Task 2.3/3.2 中 BEMT 与优化器输出的别名，两者是同一契约。"""
