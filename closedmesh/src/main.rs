#![recursion_limit = "256"]

#[tokio::main]
async fn main() {
    std::process::exit(closedmesh::run_main().await);
}
