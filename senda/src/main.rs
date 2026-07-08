#![recursion_limit = "256"]

#[tokio::main]
async fn main() {
    std::process::exit(senda::run_main().await);
}
