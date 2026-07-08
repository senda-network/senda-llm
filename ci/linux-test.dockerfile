# Full Linux build test: senda + llama.cpp (CPU/RPC)
# Run from repo root: docker build -f ci/linux-test.dockerfile -t senda-ci .
#
# NOTE: npm ci may fail behind SSL-intercepting proxies. If so, pre-build the
# UI on the host (npm run build in senda/ui/) — the dist/ is COPY'd in.
FROM rust:latest

RUN apt-get update && apt-get install -y cmake pkg-config git && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Clone llama.cpp fork (not in docker context due to .dockerignore)
RUN git clone -b rebase-upstream-master --depth 1 https://github.com/michaelneale/llama.cpp.git

# Build llama.cpp (CPU + RPC, no GPU)
RUN cmake -B llama.cpp/build -S llama.cpp \
    -DGGML_RPC=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_OPENSSL=OFF \
    && cmake --build llama.cpp/build --config Release -j$(nproc)

# Build senda (UI already built on host via npm run build, dist/ included)
COPY senda/ senda/
RUN cd senda && cargo build --release
RUN cd senda && cargo test

# Verify all binaries
RUN ls -lh senda/target/release/senda llama.cpp/build/bin/llama-server llama.cpp/build/bin/rpc-server
RUN senda/target/release/senda --version
RUN senda/target/release/senda --help | head -5
RUN llama.cpp/build/bin/llama-server --version
