# TileRT release builder / runtime image
#
# Every dep version is pinned to the validated set. Don't bump anything
# without re-running the full release pipeline (build wheel → fresh container
# → pip install wheel → pytest on B200 GPUs).
#
# Especially: transformers MUST be 4.46.3. The 5.x branch is not backward
# compatible with TileRT's tokenizer/model loading paths.
#
# The image also carries the AWS EFA userspace stack (libfabric + rdma-core);
# see the "AWS EFA userspace stack" section below for the run-time requirements,
# or build with --build-arg INSTALL_EFA=0 to leave it out.
#
# Build:
#   docker build -t tileai/tilert:cu132-v0.1.4 .
# Pull pre-built:
#   docker pull tileai/tilert:cu132-v0.1.4
# Use:
#   docker run --rm --gpus all -v $PWD:/workspace -w /workspace \
#     tileai/tilert:cu132-v0.1.4 make wheel BUILD_TYPE=Release
# Use with EFA:
#   docker run --rm --gpus all --device /dev/infiniband --ulimit memlock=-1 \
#     -v $PWD:/workspace -w /workspace tileai/tilert:cu132-v0.1.4 fi_info -p efa

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

# ── AWS EFA userspace stack ──────────────────────────────────────────────────
#
# Installs libfabric (with the EFA provider) + rdma-core into /opt/amazon/efa.
# Userspace only:
#
#   --skip-kmod        the efa kernel driver belongs to the host, not the image
#   --skip-limit-conf  memlock limits are a host/`docker run --ulimit` concern
#   --no-verify        no EFA device is present during the build
#   --mpi none         TileRT needs libfabric, not Open MPI (drop this flag if
#                      you want openmpi4/5 under /opt/amazon/openmpi*)
#
# At run time the host must have the efa kernel module loaded, and the container
# needs the devices plus unlimited memlock, e.g.:
#   docker run --gpus all --device /dev/infiniband --ulimit memlock=-1 ...
#
# Pinned by version + sha256; bump both together.
#
# Two details about running the installer here:
#  - It sources ./env.sh and ./common.sh relatively, so it must be run from its
#    own directory.
#  - Its supported-OS check matches NAME/VERSION_ID in /etc/os-release against a
#    fixed list (RHEL, Rocky, AL2023, SUSE, Debian, Ubuntu) and exits with
#    "Unsupported operating system" on anything else. This base is AlmaLinux 8,
#    which it does not recognise even though the RPMS/ROCKYLINUX8 set it ships
#    is plain el8 and ABI-correct here. The block below patches the extracted
#    (throwaway) common.sh to add AlmaLinux 8 as a supported OS and get
#    unblocked.
ARG INSTALL_EFA=1
ARG EFA_INSTALLER_VERSION=1.50.0
ARG EFA_INSTALLER_SHA256=fa6dff8593d866866c13cb4640d9059835cd4efa427971f100ab40c97bef2841
RUN if [ "${INSTALL_EFA}" = "1" ]; then \
        set -euo pipefail; \
        command -v curl >/dev/null || yum install -y --setopt=install_weak_deps=False curl; \
        archive="/tmp/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"; \
        curl -fsSL --retry 3 --retry-all-errors \
            "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz" \
            -o "${archive}"; \
        echo "${EFA_INSTALLER_SHA256}  ${archive}" | sha256sum -c -; \
        tar -xzf "${archive}" -C /tmp; \
        sed -i '/^is_rockylinux_8()/,/^}/ s/\[ "$NAME" = "Rocky Linux" \]/[ "$NAME" = "Rocky Linux" -o "$NAME" = "AlmaLinux" ]/' \
            /tmp/aws-efa-installer/common.sh; \
        grep -q '"AlmaLinux"' /tmp/aws-efa-installer/common.sh; \
        bash -n /tmp/aws-efa-installer/common.sh; \
        echo "WARNING: AlmaLinux is not on the EFA installer supported-OS list." >&2; \
        echo "WARNING: patched common.sh to accept it as Rocky Linux 8; the RPMS/ROCKYLINUX8 packages it installs are plain el8." >&2; \
        echo "NOTE: userspace install only; the EFA kernel driver and memlock limits stay with the host." >&2; \
        (cd /tmp/aws-efa-installer && \
            ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify --mpi none); \
        ldconfig; \
        test -x /opt/amazon/efa/bin/fi_info; \
        /opt/amazon/efa/bin/fi_info --version; \
        rm -rf "${archive}" /tmp/aws-efa-installer; \
        yum clean all && rm -rf /var/cache/yum; \
    fi

# fi_info and friends on PATH; the installer drops /etc/ld.so.conf.d/000_efa.conf
# so libfabric.so resolves through ldconfig without LD_LIBRARY_PATH.
ENV PATH="${PATH}:/opt/amazon/efa/bin"

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
