# hover-rotor —— 生产镜像
#
# 多阶段：builder 里编译并安装全部轮子（含 OpenCASCADE 二进制包 cadquery-ocp），
# runtime 只带 Python 运行时与 OCC 必需的 X11/OpenGL 共享库。
#
#   docker build -t hover-rotor .
#   docker run --rm -v "$PWD/results:/work/results" hover-rotor \
#       --thrust 80 --radius 0.75 --te-thickness 0.0025 --seed 2026
#
# CAD extras 默认装好：镜像存在的意义之一就是让 STEP/STL 导出可复现。
# 只跑气动主链路不需要 OCC，`pip install -e ".[dev]"` 即可。
#
# 诚实声明：本镜像**未经实机 docker build 验证**（开发机无 Docker 守护进程）。
# 达成情况见 README「验收状态」表，不要在当作已验证的前提下分发。

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build123d / ezdxf 在导入期会加载 libGL 与 glib，构建期的自检依赖它们
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# 非 editable 安装：包体进 site-packages，运行镜像因此只需带走 venv 一份
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install ".[cad,step]"

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MPLBACKEND=Agg \
    HOME=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxrender1 \
        libxext6 \
        libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /work

RUN python -c "import hover_rotor, hover_rotor.cad.step_builder, hover_rotor.cli" \
    && python -c "import build123d, ezdxf" \
    && hover-rotor-opt --help > /dev/null

ENTRYPOINT ["hover-rotor-opt"]
CMD ["--thrust", "80", "--radius", "0.75", "--te-thickness", "0.0025", "--seed", "2026"]
