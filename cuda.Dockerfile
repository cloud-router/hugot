#--- dockerfile to test hugot  ---

ARG GO_VERSION=1.25.3
ARG ONNXRUNTIME_VERSION=1.22.0
ARG GOPJRT_VERSION=0.8.3
ARG JAX_CUDA_VERSION=0.8.0
ARG BUILD_PLATFORM=linux/amd64

#--- runtime layer with all hugot dependencies for cpu and gpu ---

FROM --platform=$BUILD_PLATFORM public.ecr.aws/amazonlinux/amazonlinux:2023 AS hugot-runtime
ARG GO_VERSION
ARG ONNXRUNTIME_VERSION
ARG GOPJRT_VERSION
ARG JAX_CUDA_VERSION

ENV PATH="$PATH:/usr/local/go/bin" \
    GOPJRT_NOSUDO=1

COPY ./scripts/download-onnxruntime-gpu.sh /download-onnxruntime-gpu.sh
RUN --mount=src=./go.mod,dst=/go.mod \
    dnf --allowerasing -y install gcc jq bash tar xz gzip glibc-static libstdc++ wget zip git dirmngr sudo which && \
    ln -s /usr/lib64/libstdc++.so.6 /usr/lib64/libstdc++.so && \
    dnf install -y 'dnf-command(config-manager)' && \
    # from rhel
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo && \
    dnf install -y cuda-cudart-12-9 cuda-nvrtc-12-9 libcublas-12-9 libcurand-12-9 libcufft-12-9 libcudnn9-cuda-12 && \
    dnf clean all && \
   # go
    curl -LO https://golang.org/dl/go${GO_VERSION}.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz && \
    rm go${GO_VERSION}.linux-amd64.tar.gz && \
    # tokenizers
    tokenizer_version=$(grep 'github.com/daulet/tokenizers' /go.mod | awk '{print $2}') && \
    tokenizer_version=$(echo $tokenizer_version | awk -F'-' '{print $NF}') && \
    echo "tokenizer_version: $tokenizer_version" && \
    curl -LO https://github.com/daulet/tokenizers/releases/download/${tokenizer_version}/libtokenizers.linux-amd64.tar.gz && \
    tar -C /usr/lib -xzf libtokenizers.linux-amd64.tar.gz && \
    rm libtokenizers.linux-amd64.tar.gz && \
    # onnxruntime cpu and gpu
    sed -i 's/\r//g' /download-onnxruntime-gpu.sh && chmod +x /download-onnxruntime-gpu.sh && \
    /download-onnxruntime-gpu.sh ${ONNXRUNTIME_VERSION} && \
    # XLA/goMLX
    GOPROXY=direct go run github.com/gomlx/gopjrt/cmd/gopjrt_installer@latest -plugin=amazonlinux -version=v${GOPJRT_VERSION} -path=/usr/local && \
    GOPROXY=direct go run github.com/gomlx/gopjrt/cmd/gopjrt_installer@latest -plugin=cuda13 -version=${JAX_CUDA_VERSION} -path=/usr/local && \
    # NON-PRIVILEGED USER
    # create non-privileged testuser with id: 1000
    useradd -u 1000 -m testuser && usermod -a -G wheel testuser && \
    echo "testuser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/testuser
