# TileRT release builder / runtime image
#
# Every dep version is pinned to the validated set. Don't bump anything
# without re-running the full release pipeline (build wheel → fresh container
# → pip install wheel → pytest on B200 GPUs).
#
# Especially: transformers MUST be 4.46.3. The 5.x branch is not backward
# compatible with TileRT's tokenizer/model loading paths.
#
# Build:
#   docker build -t tileai/tilert:cu132-v0.1.4 .
# Pull pre-built:
#   docker pull tileai/tilert:cu132-v0.1.4
# Use:
#   docker run --rm --gpus all -v $PWD:/workspace -w /workspace \
#     tileai/tilert:cu132-v0.1.4 make wheel BUILD_TYPE=Release

FROM pytorch/manylinux2_28-builder:cuda13.2-main

SHELL ["/bin/bash", "-c"]

# ── System packages (glog: TileRT runtime dep; zstd: image transport) ────────
RUN yum install -y --setopt=install_weak_deps=False \
        epel-release yum-utils vim && \
    (yum config-manager --set-enabled powertools 2>/dev/null || \
     yum config-manager --set-enabled crb 2>/dev/null || true) && \
    yum --enablerepo=epel install -y --setopt=install_weak_deps=False \
        glog glog-devel zstd && \
    rpm -e --nodeps cmake 2>/dev/null || true && \
    yum clean all && rm -rf /var/cache/yum /var/tmp/* /tmp/*

# ── Conda env: python 3.12, named "tilert" ───────────────────────────────────
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create -y -n tilert python=3.12.9 && \
    conda clean -afy && rm -rf /opt/conda/pkgs/*

# ── Pinned lock set (resolved 2026-05-27 against torch 2.11.0+cu130 +
#    transformers 4.46.3 on python 3.12 / manylinux_2_28) ─────────────────────
#
# torch's METADATA transitively pins the nvidia-* cu13 runtime packages
# (cublas==13.1.0.3, cudnn-cu13==9.19.0.56, nccl-cu13==2.28.9, etc.) — those
# are NOT re-pinned here on purpose, so any patch bump in PyTorch's cu130
# release line flows through.
ARG PIP_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG PIP_EXTRA_INDEX_URL=https://pypi.org/simple
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate tilert && \
    pip install --no-cache-dir \
        --index-url "$PIP_INDEX_URL" \
        --extra-index-url "$PIP_EXTRA_INDEX_URL" \
        --upgrade pip==25.3 && \
    pip install --no-cache-dir \
        --index-url "$PIP_INDEX_URL" \
        --extra-index-url "$PIP_EXTRA_INDEX_URL" \
        "torch==2.11.0+cu130" \
        "triton==3.6.0" \
        "transformers==4.46.3" \
        "tokenizers==0.20.3" \
        "huggingface_hub==0.35.3" \
        "hf_xet==1.1.10" \
        "safetensors==0.6.2" \
        "regex==2025.9.18" \
        "requests==2.32.3" \
        "charset_normalizer==3.3.2" \
        "idna==3.7" \
        "urllib3==2.3.0" \
        "certifi==2026.2.25" \
        "packaging==24.2" \
        "tqdm==4.67.1" \
        "pyyaml==6.0.2" \
        "numpy==2.3.2" \
        "einops==0.8.1" \
        "filelock==3.29.0" \
        "fsspec==2026.4.0" \
        "jinja2==3.1.6" \
        "MarkupSafe==3.0.3" \
        "networkx==3.6.1" \
        "sympy==1.14.0" \
        "mpmath==1.3.0" \
        "typing_extensions==4.15.0" \
        "setuptools==81.0.0" \
        "importlib_metadata==8.7.1" \
        "zipp==3.23.0" \
        "scikit-build-core==0.12.2" \
        "setuptools-scm==9.2.2" \
        "vcs-versioning==1.1.1" \
        "pathspec==1.1.1" \
        "ninja==1.13.0" \
        "cmake==4.1.2" \
        "pytest==8.4.1" \
        "pytest-cov==7.1.0" \
        "pluggy==1.6.0" \
        "iniconfig==2.3.0" \
        "pygments==2.20.0" \
        "tomli==2.4.1" \
        "coverage==7.10.7" \
        "exceptiongroup==1.3.1" && \
    python -c 'import torch, triton, transformers, tokenizers; assert torch.__version__ == "2.11.0+cu130", torch.__version__; assert torch.version.cuda.startswith("13"), torch.version.cuda; assert triton.__version__ == "3.6.0", triton.__version__; assert transformers.__version__ == "4.46.3", transformers.__version__; assert tokenizers.__version__ == "0.20.3", tokenizers.__version__; print("torch", torch.__version__, "cuda", torch.version.cuda, "| triton", triton.__version__, "| transformers", transformers.__version__, "| tokenizers", tokenizers.__version__, "OK")' && \
    pip cache purge && rm -rf /root/.cache/pip /root/.cache/* && \
    conda clean -afy && \
    find /opt/conda -type f -name "*.pyc" -delete && \
    find /opt/conda -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# ── CUDA arch (Blackwell sm_100) + scikit-build pass-through ─────────────────
ENV TORCH_CUDA_ARCH_LIST="10.0" \
    CUDAARCHS="100" \
    CMAKE_ARGS="-DUSER_CUDA_ARCH_LIST=10.0" \
    SKBUILD_CMAKE_DEFINE="USER_CUDA_ARCH_LIST=10.0" \
    CMAKE_BUILD_PARALLEL_LEVEL=16 \
    PATH="/opt/conda/envs/tilert/bin:/opt/conda/bin:${PATH}"

# ── Shell activation + entrypoint ─────────────────────────────────────────────
RUN { echo 'export PATH=/opt/conda/envs/tilert/bin:/opt/conda/bin:$PATH'; \
      echo '. /opt/conda/etc/profile.d/conda.sh'; \
      echo 'conda activate tilert 2>/dev/null || true'; \
    } >> /etc/bashrc && \
    printf '%s\n' \
        '#!/bin/bash' \
        'set -e' \
        '. /opt/conda/etc/profile.d/conda.sh' \
        'conda activate tilert' \
        'exec "$@"' \
        > /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
