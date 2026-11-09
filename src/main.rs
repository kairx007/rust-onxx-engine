use clap::Parser;
use rust_engine::cli::Cli;

fn main() {
    if let Err(error) = Cli::parse().execute() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}
