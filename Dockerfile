FROM pytorch/manylinux2_28-builder:cuda12.9-main

SHELL ["/bin/bash", "-c"]

RUN yum update -y && \
    yum install -y epel-release yum-utils vim && \
    (yum config-manager --set-enabled powertools || \
     yum config-manager --set-enabled crb || true) && \
    yum --enablerepo=epel install -y glog glog-devel && \
    yum clean all && \
    rm -rf /var/cache/yum /var/tmp/* /tmp/*

RUN conda init bash && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda create -y -n tilert python=3.12 && \
    conda activate tilert && \
    conda clean -afy && \
    rm -rf /opt/conda/pkgs/* /opt/conda/conda-meta/*.json.bak

COPY requirements.txt requirements-dev.txt /tmp/
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate tilert && \
    pip install --no-cache-dir -r /tmp/requirements-dev.txt && \
    pip cache purge && \
    rm -rf /tmp/requirements*.txt /root/.cache/pip /root/.cache/* && \
    conda clean -afy && \
    find /opt/conda -type f -name "*.pyc" -delete && \
    find /opt/conda -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

RUN echo "alias cls='clear'" >> ~/.bashrc && \
    echo "alias ll='ls -l'" >> ~/.bashrc && \
    echo "alias la='ls -a'" >> ~/.bashrc && \
    echo "alias vi='vim'" >> ~/.bashrc && \
    echo "alias grep='grep --color=auto'" >> ~/.bashrc && \
    echo "export PATH=\"/opt/conda/bin:\$PATH\"" >> ~/.bashrc && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate tilert" >> ~/.bashrc && \
    echo '#!/bin/bash' > /usr/local/bin/entrypoint.sh && \
    echo 'export PATH="/opt/conda/bin:$PATH"' >> /usr/local/bin/entrypoint.sh && \
    echo '. /opt/conda/etc/profile.d/conda.sh' >> /usr/local/bin/entrypoint.sh && \
    echo 'conda activate tilert' >> /usr/local/bin/entrypoint.sh && \
    echo 'exec "$@"' >> /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
