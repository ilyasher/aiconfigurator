fn main() {
    if let Err(err) = aic_rust::cli::run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}
