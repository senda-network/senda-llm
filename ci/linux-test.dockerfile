# Full Linux build test: closedmesh + llama.cpp (CPU/RPC)
# Run from repo root: docker build -f ci/linux-test.dockerfile -t closedmesh-ci .
#
# NOTE: npm ci may fail behind SSL-intercepting proxies. If so, pre-build the
# UI on the host (npm run build in closedmesh/ui/) — the dist/ is COPY'd in.
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

# Build closedmesh (UI already built on host via npm run build, dist/ included)
COPY closedmesh/ closedmesh/
RUN cd closedmesh && cargo build --release
RUN cd closedmesh && cargo test

# Verify all binaries
RUN ls -lh closedmesh/target/release/closedmesh llama.cpp/build/bin/llama-server llama.cpp/build/bin/rpc-server
RUN closedmesh/target/release/closedmesh --version
RUN closedmesh/target/release/closedmesh --help | head -5
RUN llama.cpp/build/bin/llama-server --version
