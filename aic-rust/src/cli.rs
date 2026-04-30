use crate::estimator::{
    run_native_default, AggResult, NativeDefaultRequest, NativeDefaultResult, ParetoPoint,
};
use crate::model::{estimate_model_weight_bytes, load_model_info};
use crate::paths::default_systems_root;
use crate::perf_database::PerfDatabase;
use crate::support::check_support;
use crate::system::{latest_database_version, SystemSpec};
use crate::types::{
    BackendName, DatabaseMode, FmhaQuantMode, GemmQuantMode, KvCacheQuantMode, MoeQuantMode,
};
use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use serde::Serialize;
use std::ffi::OsString;
use std::str::FromStr;

#[derive(Debug, Parser)]
#[command(name = "aic-rust")]
#[command(about = "Rust rewrite-in-progress for aiconfigurator SDK")]
pub struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    #[command(subcommand)]
    Cli(CliMode),
    QueryGemm(QueryGemmArgs),
    QueryContextAttention(QueryContextAttentionArgs),
    QueryGenerationAttention(QueryGenerationAttentionArgs),
    QueryMoe(QueryMoeArgs),
    ModelInfo(ModelInfoArgs),
}

#[derive(Debug, Subcommand)]
enum CliMode {
    Support(SupportArgs),
    Generate(GenerateArgs),
    Default(DefaultArgs),
}

#[derive(Debug, Parser)]
struct SupportArgs {
    #[arg(long, alias = "model")]
    model_path: String,
    #[arg(long)]
    system: String,
    #[arg(long)]
    backend: Option<BackendName>,
    #[arg(long)]
    backend_version: Option<String>,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Parser)]
struct GenerateArgs {
    #[arg(long, alias = "model")]
    model_path: String,
    #[arg(long)]
    total_gpus: u32,
    #[arg(long)]
    system: String,
    #[arg(long, default_value = "trtllm")]
    backend: BackendName,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Parser)]
struct DefaultArgs {
    #[arg(long, alias = "model")]
    model_path: String,
    #[arg(long)]
    total_gpus: u32,
    #[arg(long)]
    system: String,
    #[arg(long, default_value = "trtllm")]
    backend: BackendName,
}

#[derive(Debug, Parser)]
struct QueryGemmArgs {
    #[arg(long)]
    system: String,
    #[arg(long, default_value = "trtllm")]
    backend: BackendName,
    #[arg(long)]
    version: Option<String>,
    #[arg(long)]
    m: i64,
    #[arg(long)]
    n: i64,
    #[arg(long)]
    k: i64,
    #[arg(long)]
    quant: GemmQuantMode,
    #[arg(long, default_value = "SILICON")]
    database_mode: DatabaseMode,
}

#[derive(Debug, Parser)]
struct QueryContextAttentionArgs {
    #[arg(long)]
    system: String,
    #[arg(long, default_value = "trtllm")]
    backend: BackendName,
    #[arg(long)]
    version: Option<String>,
    #[arg(long)]
    batch_size: i64,
    #[arg(long)]
    seq_len: i64,
    #[arg(long, default_value_t = 0)]
    prefix: i64,
    #[arg(long)]
    num_heads: i64,
    #[arg(long)]
    num_kv_heads: i64,
    #[arg(long)]
    kv_cache_quant: KvCacheQuantMode,
    #[arg(long)]
    fmha_quant: FmhaQuantMode,
    #[arg(long, default_value_t = 0)]
    window_size: i64,
    #[arg(long, default_value_t = 128)]
    head_size: i64,
    #[arg(long, default_value = "SILICON")]
    database_mode: DatabaseMode,
}

#[derive(Debug, Parser)]
struct QueryGenerationAttentionArgs {
    #[arg(long)]
    system: String,
    #[arg(long, default_value = "trtllm")]
    backend: BackendName,
    #[arg(long)]
    version: Option<String>,
    #[arg(long)]
    batch_size: i64,
    #[arg(long)]
    seq_len: i64,
    #[arg(long)]
    num_heads: i64,
    #[arg(long)]
    num_kv_heads: i64,
    #[arg(long)]
    kv_cache_quant: KvCacheQuantMode,
    #[arg(long, default_value_t = 0)]
    window_size: i64,
    #[arg(long, default_value_t = 128)]
    head_size: i64,
    #[arg(long, default_value = "SILICON")]
    database_mode: DatabaseMode,
}

#[derive(Debug, Parser)]
struct QueryMoeArgs {
    #[arg(long)]
    system: String,
    #[arg(long, default_value = "trtllm")]
    backend: BackendName,
    #[arg(long)]
    version: Option<String>,
    #[arg(long)]
    num_tokens: i64,
    #[arg(long)]
    hidden_size: i64,
    #[arg(long)]
    inter_size: i64,
    #[arg(long)]
    topk: i64,
    #[arg(long)]
    num_experts: i64,
    #[arg(long)]
    moe_tp_size: u32,
    #[arg(long)]
    moe_ep_size: u32,
    #[arg(long)]
    quant: MoeQuantMode,
    #[arg(long, default_value = "uniform")]
    workload_distribution: String,
    #[arg(long, default_value_t = true)]
    is_context: bool,
    #[arg(long)]
    moe_backend: Option<String>,
    #[arg(long, default_value = "SILICON")]
    database_mode: DatabaseMode,
    #[arg(long, default_value_t = true)]
    is_gated: bool,
    #[arg(long, default_value_t = false)]
    enable_eplb: bool,
}

#[derive(Debug, Parser)]
struct ModelInfoArgs {
    #[arg(long, alias = "model")]
    model_path: String,
}

#[derive(Debug, Serialize)]
struct GenerateResult {
    model: String,
    system: String,
    backend: String,
    total_gpus: u32,
    gpus_used: u32,
    tensor_parallel_size: u32,
    pipeline_parallel_size: u32,
    replicas: u32,
    estimated_model_weight_bytes: u128,
}

pub fn run() -> Result<()> {
    let raw_args: Vec<OsString> = std::env::args_os().collect();
    if is_cli_default_invocation(&raw_args) {
        return cmd_default_passthrough(&raw_args);
    }

    let args = Args::parse();
    match args.command {
        Command::Cli(CliMode::Support(args)) => cmd_support(args),
        Command::Cli(CliMode::Generate(args)) => cmd_generate(args),
        Command::Cli(CliMode::Default(args)) => cmd_default(args),
        Command::QueryGemm(args) => cmd_query_gemm(args),
        Command::QueryContextAttention(args) => cmd_query_context_attention(args),
        Command::QueryGenerationAttention(args) => cmd_query_generation_attention(args),
        Command::QueryMoe(args) => cmd_query_moe(args),
        Command::ModelInfo(args) => cmd_model_info(args),
    }
}

fn cmd_support(args: SupportArgs) -> Result<()> {
    let result = check_support(
        &args.model_path,
        &args.system,
        args.backend,
        args.backend_version.as_deref(),
    )?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("Aggregated Support: {}", yes_no(result.agg_supported));
        println!("Disaggregated Support: {}", yes_no(result.disagg_supported));
    }
    Ok(())
}

fn cmd_generate(args: GenerateArgs) -> Result<()> {
    let result = generate_naive(
        &args.model_path,
        args.total_gpus,
        &args.system,
        args.backend,
    )?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("Naive Configuration Generated Successfully");
        println!("Model: {}", result.model);
        println!("System: {}", result.system);
        println!("Backend: {}", result.backend);
        println!(
            "Total GPUs: {} (using {})",
            result.total_gpus, result.gpus_used
        );
        println!("Parallelism: TP={}, PP=1", result.tensor_parallel_size);
        println!("Replicas: {}", result.replicas);
    }
    Ok(())
}

fn cmd_default(args: DefaultArgs) -> Result<()> {
    let _ = args;
    bail!(
        "Rust cli default should have been handled by the compatibility executor before clap parsing."
    )
}

fn is_cli_default_invocation(args: &[OsString]) -> bool {
    args.get(1).and_then(|value| value.to_str()) == Some("cli")
        && args
            .iter()
            .skip(2)
            .any(|value| value.to_str() == Some("default"))
}

fn cmd_default_passthrough(args: &[OsString]) -> Result<()> {
    if args
        .iter()
        .any(|arg| arg.to_str() == Some("--help") || arg.to_str() == Some("-h"))
    {
        print_native_default_help();
        return Ok(());
    }
    let json = args.iter().any(|arg| arg.to_str() == Some("--json"));
    let parsed = parse_native_default_request(args)?;
    let report = if parsed.backend_auto {
        run_native_default_auto(&parsed.req)?
    } else {
        run_native_default(&parsed.req)?
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_native_default_report(&report);
    }
    Ok(())
}

struct ParsedNativeDefaultRequest {
    req: NativeDefaultRequest,
    backend_auto: bool,
}

fn parse_native_default_request(args: &[OsString]) -> Result<ParsedNativeDefaultRequest> {
    let default_idx = args
        .iter()
        .position(|arg| arg.to_str() == Some("default"))
        .context("missing default subcommand")?;
    let mut model_path = None;
    let mut total_gpus = None;
    let mut system = None;
    let mut backend = BackendName::trtllm;
    let mut backend_auto = false;
    let mut backend_version = None;
    let mut database_mode = DatabaseMode::SILICON;
    let mut isl = 4000_i64;
    let mut osl = 1000_i64;
    let mut ttft = 2000.0_f64;
    let mut tpot = 30.0_f64;
    let mut request_latency = None;
    let mut prefix = 0_i64;
    let mut top_n = 5_usize;
    let mut free_gpu_memory_fraction = Some(1.0_f64);
    let mut max_seq_len = None;
    let mut enable_chunked_prefill = false;

    let mut iter = args.iter().skip(default_idx + 1).peekable();
    while let Some(arg) = iter.next() {
        let Some(raw) = arg.to_str() else {
            bail!("non-utf8 CLI argument is not supported")
        };
        let (key, inline_value) = match raw.split_once('=') {
            Some((key, value)) if key.starts_with("--") => (key, Some(value.to_string())),
            _ => (raw, None),
        };
        let mut next_value = || -> Result<String> {
            if let Some(value) = inline_value.clone() {
                return Ok(value);
            }
            iter.next()
                .and_then(|v| v.to_str())
                .map(ToString::to_string)
                .with_context(|| format!("missing value for {key}"))
        };
        match key {
            "--model-path" | "--model" => model_path = Some(next_value()?),
            "--total-gpus" => total_gpus = Some(next_value()?.parse()?),
            "--system" => system = Some(next_value()?),
            "--decode-system" => {
                let value = next_value()?;
                if Some(value.as_str()) != system.as_deref() {
                    bail!("native default does not support heterogeneous decode-system yet");
                }
            }
            "--backend" => {
                let value = next_value()?;
                if value == "auto" {
                    backend_auto = true;
                    backend = BackendName::trtllm;
                } else {
                    backend_auto = false;
                    backend = BackendName::from_str(&value)?;
                }
            }
            "--backend-version" => backend_version = Some(next_value()?),
            "--database-mode" => database_mode = DatabaseMode::from_str(&next_value()?)?,
            "--isl" => isl = next_value()?.parse()?,
            "--osl" => osl = next_value()?.parse()?,
            "--ttft" => ttft = next_value()?.parse()?,
            "--tpot" => tpot = next_value()?.parse()?,
            "--request-latency" => request_latency = Some(next_value()?.parse()?),
            "--prefix" => prefix = next_value()?.parse()?,
            "--top-n" => top_n = next_value()?.parse()?,
            "--free-gpu-memory-fraction" => free_gpu_memory_fraction = Some(next_value()?.parse()?),
            "--max-seq-len" => max_seq_len = Some(next_value()?.parse()?),
            "--no-color" | "--strict-sla" | "--debug" | "--json" => {}
            "--save-dir"
            | "--systems-paths"
            | "--deployment-target"
            | "--generator-config"
            | "--generator-dynamo-version"
            | "--generator-set" => {
                let _ = next_value()?;
                bail!("native default does not support generator/save options yet");
            }
            "--enable-chunked-prefill" => enable_chunked_prefill = true,
            "--enable-wideep" => {
                bail!("native default does not support {key} yet");
            }
            other if other.starts_with("--top-") => {
                top_n = other
                    .trim_start_matches("--top-")
                    .parse()
                    .with_context(|| format!("invalid top-N shortcut: {other}"))?;
            }
            other => bail!("unsupported native default argument: {other}"),
        }
    }

    Ok(ParsedNativeDefaultRequest {
        req: NativeDefaultRequest {
            model_path: model_path.context("--model-path is required")?,
            total_gpus: total_gpus.context("--total-gpus is required")?,
            system: system.context("--system is required")?,
            backend,
            backend_version,
            database_mode,
            isl,
            osl,
            ttft,
            tpot,
            request_latency,
            prefix,
            top_n,
            free_gpu_memory_fraction,
            max_seq_len,
            enable_chunked_prefill,
        },
        backend_auto,
    })
}

fn run_native_default_auto(
    req: &NativeDefaultRequest,
) -> Result<crate::estimator::NativeDefaultReport> {
    let mut best = None;
    let mut errors = Vec::new();
    for backend in [BackendName::trtllm, BackendName::vllm, BackendName::sglang] {
        let mut backend_req = req.clone();
        backend_req.backend = backend;
        match run_native_default(&backend_req) {
            Ok(report) => {
                if best
                    .as_ref()
                    .is_none_or(|current: &crate::estimator::NativeDefaultReport| {
                        report.best_throughput > current.best_throughput
                    })
                {
                    best = Some(report);
                }
            }
            Err(err) => errors.push(format!("{backend}: {err:#}")),
        }
    }
    best.with_context(|| {
        format!(
            "native default --backend auto found no supported backend: {}",
            errors.join("; ")
        )
    })
}

fn print_native_default_help() {
    println!("Run the native Rust default agg vs disagg comparison");
    println!();
    println!("Usage: aic-rust cli default --model-path <MODEL> --total-gpus <N> --system <SYSTEM> [OPTIONS]");
    println!();
    println!("Options:");
    println!("      --model-path, --model <MODEL>");
    println!("      --total-gpus <N>");
    println!("      --system <SYSTEM>");
    println!("      --decode-system <SYSTEM>");
    println!("      --backend <trtllm|vllm|sglang|auto>");
    println!("      --backend-version <VERSION>");
    println!("      --database-mode <SILICON|HYBRID|EMPIRICAL|SOL>");
    println!("      --isl <TOKENS>");
    println!("      --osl <TOKENS>");
    println!("      --ttft <MS>");
    println!("      --tpot <MS>");
    println!("      --request-latency <MS>");
    println!("      --prefix <TOKENS>");
    println!("      --top-n <N>");
    println!("      --free-gpu-memory-fraction <FRACTION>");
    println!("      --max-seq-len <TOKENS>");
    println!("      --enable-chunked-prefill");
    println!("      --enable-wideep");
    println!("      --json");
    println!("      --strict-sla");
    println!("      --no-color");
    println!("  -h, --help");
}

fn print_native_default_report(report: &crate::estimator::NativeDefaultReport) {
    let best = best_metrics(report);
    println!("{}", "*".repeat(80));
    println!("*{:^78}*", " AIConfigurator Final Results ");
    println!("{}", "*".repeat(80));
    println!("  {}", "-".repeat(76));
    println!("  Input Configuration & SLA Target:");
    println!(
        "    Model: {} (is_moe: {})",
        report.model_path,
        if report.is_moe { "True" } else { "False" }
    );
    println!("    Total GPUs: {}", report.total_gpus);
    println!(
        "    Best Experiment Chosen: {} at {:.2} tokens/s/gpu (disagg {:.2}x better)",
        report.chosen_exp, report.best_throughput, report.disagg_ratio
    );
    println!("  {}", "-".repeat(76));
    println!("  Overall Best Configuration:");
    println!(
        "    - Best Throughput: {} tokens/s",
        format_number(best.tokens_s, 2)
    );
    println!(
        "    - Per-GPU Throughput: {:.2} tokens/s/gpu",
        report.best_throughput
    );
    println!(
        "    - Per-User Throughput: {:.2} tokens/s/user",
        best.tokens_s_user
    );
    println!("    - Request Rate: {:.2} req/s", best.request_rate);
    println!("    - TTFT: {:.2}ms", best.ttft);
    println!("    - TPOT: {:.2}ms", best.tpot);
    println!("    - Request Latency: {:.2}ms", best.request_latency);
    println!("  {}", "-".repeat(76));
    if let Some(chart) = render_pareto_chart(report) {
        println!("  Pareto Frontier:");
        println!("{chart}");
    }
    println!("  {}", "-".repeat(76));
    println!("  Deployment Details:");
    println!(
        "    (p) stands for prefill, (d) stands for decode, bs stands for batch size, a replica stands for the smallest scalable unit xPyD of the disagg system"
    );
    println!("    Some math: total gpus used = replicas * gpus/replica");
    println!(
        "               gpus/replica = (p)gpus/worker * (p)workers + (d)gpus/worker * (d)workers; for Agg, gpus/replica = gpus/worker"
    );
    println!(
        "               gpus/worker = tp * pp * dp = etp * ep * pp for MoE models; tp * pp for dense models (underlined numbers are the actual values in math)"
    );

    if !report.top_agg_configs.is_empty() {
        println!();
        println!("{}", render_agg_top_table(report));
    }
    if !report.top_disagg_configs.is_empty() {
        println!();
        println!("{}", render_disagg_top_table(report));
    }
    println!("{}", "*".repeat(80));
}

struct ParetoSeries<'a> {
    label: &'a str,
    marker: char,
    points: &'a [ParetoPoint],
}

struct BestMetrics {
    tokens_s: f64,
    tokens_s_user: f64,
    request_rate: f64,
    ttft: f64,
    tpot: f64,
    request_latency: f64,
}

fn best_metrics(report: &crate::estimator::NativeDefaultReport) -> BestMetrics {
    match &report.best {
        NativeDefaultResult::Agg(row) => {
            let replicas = report.total_gpus / row.num_total_gpus;
            BestMetrics {
                tokens_s: row.tokens_s * replicas as f64,
                tokens_s_user: row.tokens_s_user,
                request_rate: row.request_rate * replicas as f64,
                ttft: row.ttft,
                tpot: row.tpot,
                request_latency: row.request_latency,
            }
        }
        NativeDefaultResult::Disagg(row) => {
            let replicas = report.total_gpus / row.num_total_gpus;
            BestMetrics {
                tokens_s: row.tokens_s * replicas as f64,
                tokens_s_user: row.tokens_s_user,
                request_rate: row.request_rate * replicas as f64,
                ttft: row.ttft,
                tpot: row.tpot,
                request_latency: row.request_latency,
            }
        }
    }
}

fn render_pareto_chart(report: &crate::estimator::NativeDefaultReport) -> Option<String> {
    let mut series = Vec::new();
    if !report.agg_pareto_front.is_empty() {
        series.push(ParetoSeries {
            label: "agg",
            marker: '*',
            points: &report.agg_pareto_front,
        });
    }
    if !report.disagg_pareto_front.is_empty() {
        series.push(ParetoSeries {
            label: "disagg",
            marker: 'f',
            points: &report.disagg_pareto_front,
        });
    }
    if series.is_empty() {
        return None;
    }

    let best = best_pareto_point(report);
    let mut x_min = f64::INFINITY;
    let mut x_max = 0.0_f64;
    let mut y_max = 0.0_f64;
    for item in &series {
        for point in item.points {
            x_min = x_min.min(point.x);
            x_max = x_max.max(point.x);
            y_max = y_max.max(point.y);
        }
    }
    if let Some(point) = &best {
        x_min = x_min.min(point.x);
        x_max = x_max.max(point.x);
        y_max = y_max.max(point.y);
    }
    if !(x_max > 0.0 && y_max > 0.0) {
        return None;
    }

    y_max = ceil_to(y_max * 1.2, 100.0);
    let x_limit = ((x_max * 1.1 + 19.0) / 20.0).floor() * 20.0;
    let has_points_within_cap = x_min <= 300.0;
    x_max = if has_points_within_cap {
        x_limit.min(300.0)
    } else {
        x_limit
    };
    if !(x_max > 0.0 && y_max > 0.0) {
        return None;
    }

    let width = 71_usize;
    let height = 18_usize;
    let mut grid = vec![vec![' '; width]; height];

    for item in &series {
        let mut previous = None;
        for point in item.points {
            let current = map_point(point, x_max, y_max, width, height);
            if let (Some((prev_row, prev_col)), Some((row, col))) = (previous, current) {
                draw_line(&mut grid, prev_row, prev_col, row, col, item.marker);
            }
            previous = current;
        }
        for point in item.points {
            if let Some((row, col)) = map_point(point, x_max, y_max, width, height) {
                plot_char(&mut grid, row, col, item.marker);
            }
        }
    }

    if let Some(point) = &best {
        if let Some((row, col)) = map_point(point, x_max, y_max, width, height) {
            grid[row][col] = 'x';
        }
    }

    overlay_legend(&mut grid, &series, &report.chosen_exp);

    let mut out = String::new();
    out.push_str(&format!(
        "{:^80}\n",
        format!(
            "{} Pareto Frontier: tokens/s/gpu_cluster vs {}",
            report.model_path, report.pareto_x_axis
        )
    ));
    out.push_str(&format!("       +{}+\n", "-".repeat(width)));
    let ticks = y_ticks(y_max, height);
    for (row_idx, row) in grid.iter().enumerate() {
        if let Some(value) = ticks.iter().find_map(|(tick_row, value)| {
            if *tick_row == row_idx {
                Some(*value)
            } else {
                None
            }
        }) {
            out.push_str(&format!(
                "{:>7}+{}|\n",
                format!("{value:.1}"),
                row.iter().collect::<String>()
            ));
        } else {
            out.push_str(&format!("       |{}|\n", row.iter().collect::<String>()));
        }
    }
    out.push_str(&x_axis_border(width));
    out.push_str(&x_tick_line(x_max, width));
    out.push('\n');
    out.push_str(&format!(
        "{:<40}{}\n",
        "tokens/s/gpu_cluster", report.pareto_x_axis
    ));
    Some(out)
}

fn best_pareto_point(report: &crate::estimator::NativeDefaultReport) -> Option<ParetoPoint> {
    let use_request_latency = report.pareto_x_axis == "request_latency";
    match &report.best {
        NativeDefaultResult::Agg(row) => Some(ParetoPoint {
            x: if use_request_latency {
                row.request_latency
            } else {
                row.tokens_s_user
            },
            y: row.tokens_s_gpu_cluster,
        }),
        NativeDefaultResult::Disagg(row) => Some(ParetoPoint {
            x: if use_request_latency {
                row.request_latency
            } else {
                row.tokens_s_user
            },
            y: row.tokens_s_gpu_cluster,
        }),
    }
}

fn map_point(
    point: &ParetoPoint,
    x_max: f64,
    y_max: f64,
    width: usize,
    height: usize,
) -> Option<(usize, usize)> {
    if point.x < 0.0 || point.x > x_max || point.y < 0.0 || point.y > y_max {
        return None;
    }
    let col = ((point.x / x_max) * (width - 1) as f64).round() as usize;
    let row = height - 1 - ((point.y / y_max) * (height - 1) as f64).round() as usize;
    Some((row.min(height - 1), col.min(width - 1)))
}

fn draw_line(
    grid: &mut [Vec<char>],
    row0: usize,
    col0: usize,
    row1: usize,
    col1: usize,
    marker: char,
) {
    let mut row = row0 as isize;
    let mut col = col0 as isize;
    let end_row = row1 as isize;
    let end_col = col1 as isize;
    let d_col = (end_col - col).abs();
    let d_row = -(end_row - row).abs();
    let step_col = if col < end_col { 1 } else { -1 };
    let step_row = if row < end_row { 1 } else { -1 };
    let mut err = d_col + d_row;

    loop {
        let cell = &mut grid[row as usize][col as usize];
        if *cell == ' ' {
            *cell = marker;
        }
        if row == end_row && col == end_col {
            break;
        }
        let e2 = 2 * err;
        if e2 >= d_row {
            err += d_row;
            col += step_col;
        }
        if e2 <= d_col {
            err += d_col;
            row += step_row;
        }
    }
}

fn plot_char(grid: &mut [Vec<char>], row: usize, col: usize, marker: char) {
    let cell = &mut grid[row][col];
    *cell = match *cell {
        ' ' | '.' => marker,
        existing if existing == marker => existing,
        _ => '#',
    };
}

fn x_tick_line(x_max: f64, width: usize) -> String {
    let mut chars = vec![' '; width + 9];
    for idx in 0..=4 {
        let value = x_max * idx as f64 / 4.0;
        let label = format_axis_value(value);
        let col = 8 + ((width - 1) as f64 * idx as f64 / 4.0).round() as usize;
        place_label(&mut chars, col.saturating_sub(label.len() / 2), &label);
    }
    chars.into_iter().collect()
}

fn place_label(chars: &mut [char], start: usize, label: &str) {
    for (idx, ch) in label.chars().enumerate() {
        let pos = start + idx;
        if pos < chars.len() {
            chars[pos] = ch;
        }
    }
}

fn format_axis_value(value: f64) -> String {
    if (value.round() - value).abs() < 1e-9 {
        format!("{value:.0}")
    } else if value.abs() >= 1000.0 {
        format!("{value:.0}")
    } else if value.abs() >= 100.0 {
        format!("{value:.1}")
    } else if value.abs() >= 10.0 {
        format!("{value:.2}")
    } else {
        format!("{value:.3}")
    }
}

fn ceil_to(value: f64, step: f64) -> f64 {
    if value <= 0.0 {
        return 0.0;
    }
    (value / step).ceil() * step
}

fn y_ticks(y_max: f64, height: usize) -> Vec<(usize, f64)> {
    (0..=6)
        .map(|idx| {
            let row = ((height - 1) as f64 * idx as f64 / 6.0).round() as usize;
            let value = y_max * (6 - idx) as f64 / 6.0;
            (row, value)
        })
        .collect()
}

fn overlay_legend(grid: &mut [Vec<char>], series: &[ParetoSeries<'_>], chosen_exp: &str) {
    for (row, item) in series.iter().enumerate() {
        write_grid_text(
            grid,
            row,
            1,
            &format!("{}{} {}", item.marker, item.marker, item.label),
        );
    }
    write_grid_text(grid, series.len(), 1, &format!("xx {chosen_exp} best"));
}

fn write_grid_text(grid: &mut [Vec<char>], row: usize, col: usize, text: &str) {
    if row >= grid.len() {
        return;
    }
    for (idx, ch) in text.chars().enumerate() {
        let target = col + idx;
        if target < grid[row].len() {
            grid[row][target] = ch;
        }
    }
}

fn x_axis_border(width: usize) -> String {
    let mut chars = vec!['-'; width];
    for idx in 0..=4 {
        let col = ((width - 1) as f64 * idx as f64 / 4.0).round() as usize;
        if col < chars.len() {
            chars[col] = '+';
        }
    }
    format!("       +{}+\n", chars.into_iter().collect::<String>())
}

fn render_agg_top_table(report: &crate::estimator::NativeDefaultReport) -> String {
    let mut rows = Vec::new();
    for (idx, row) in report.top_agg_configs.iter().enumerate() {
        let replicas = report.total_gpus / row.num_total_gpus;
        let total_used = replicas * row.num_total_gpus;
        let worker_gpus = if report.is_moe {
            row.pp * row.tp * row.dp
        } else {
            row.pp * row.tp
        };
        rows.push(vec![
            (idx + 1).to_string(),
            row.backend.clone(),
            format!("{:.2}", row.tokens_s_gpu_cluster),
            format!("{:.2}", row.tokens_s_user),
            format!("{:.2}", row.request_rate * replicas as f64),
            format!("{:.2}", row.ttft),
            format!("{:.2}", row.request_latency),
            format!(
                "{} (={}x{})",
                format_count(row.concurrency * replicas as f64),
                format_count(row.concurrency),
                replicas
            ),
            format!(
                "{} ({}={}x{})",
                report.total_gpus, total_used, replicas, row.num_total_gpus
            ),
            replicas.to_string(),
            row.num_total_gpus.to_string(),
            agg_gpus_worker(row, report.is_moe, worker_gpus),
            agg_parallel(row, report.is_moe),
            row.bs.to_string(),
        ]);
    }
    let headers = [
        "Rank",
        "backend",
        "tokens/s/gpu",
        "tokens/s/user",
        "req/s",
        "TTFT",
        "request_latency",
        "concurrency",
        "total_gpus (used)",
        "replicas",
        "gpus/replica",
        "gpus/worker",
        "parallel",
        "bs",
    ];
    format!(
        "agg Top Configurations: (Sorted by tokens/s/gpu)\n{}",
        render_table(&headers, &rows)
    )
}

fn render_disagg_top_table(report: &crate::estimator::NativeDefaultReport) -> String {
    let mut rows = Vec::new();
    for (idx, row) in report.top_disagg_configs.iter().enumerate() {
        let replicas = report.total_gpus / row.num_total_gpus;
        let total_used = replicas * row.num_total_gpus;
        let p_worker_gpus = row.prefill_pp * row.prefill_tp * row.prefill_dp;
        let d_worker_gpus = row.decode_pp * row.decode_tp * row.decode_dp;
        rows.push(vec![
            (idx + 1).to_string(),
            row.backend.clone(),
            format!("{:.2}", row.tokens_s_gpu_cluster),
            format!("{:.2}", row.tokens_s_user),
            format!("{:.2}", row.request_rate * replicas as f64),
            format!("{:.2}", row.ttft),
            format!("{:.2}", row.request_latency),
            format!(
                "{} (={}x{})",
                format_count(row.concurrency * replicas as f64),
                format_count(row.concurrency),
                replicas
            ),
            format!(
                "{} ({}={}x{})",
                report.total_gpus, total_used, replicas, row.num_total_gpus
            ),
            replicas.to_string(),
            format!(
                "{} (={}x{}+{}x{})",
                row.num_total_gpus,
                row.prefill_workers,
                p_worker_gpus,
                row.decode_workers,
                d_worker_gpus
            ),
            row.prefill_workers.to_string(),
            disagg_gpus_worker(
                row.prefill_tp,
                row.prefill_pp,
                row.prefill_dp,
                report.is_moe,
            ),
            disagg_parallel(
                row.prefill_tp,
                row.prefill_pp,
                row.prefill_dp,
                1,
                1,
                report.is_moe,
            ),
            row.prefill_bs.to_string(),
            row.decode_workers.to_string(),
            disagg_gpus_worker(row.decode_tp, row.decode_pp, row.decode_dp, report.is_moe),
            disagg_parallel(
                row.decode_tp,
                row.decode_pp,
                row.decode_dp,
                1,
                1,
                report.is_moe,
            ),
            row.decode_bs.to_string(),
        ]);
    }
    let headers = [
        "Rank",
        "backend",
        "tokens/s/gpu",
        "tokens/s/user",
        "req/s",
        "TTFT",
        "request_latency",
        "concurrency",
        "total_gpus (used)",
        "replicas",
        "gpus/replica",
        "(p)workers",
        "(p)gpus/worker",
        "(p)parallel",
        "(p)bs",
        "(d)workers",
        "(d)gpus/worker",
        "(d)parallel",
        "(d)bs",
    ];
    format!(
        "disagg Top Configurations: (Sorted by tokens/s/gpu)\n{}",
        render_table(&headers, &rows)
    )
}

fn agg_gpus_worker(row: &AggResult, is_moe: bool, worker_gpus: u32) -> String {
    if is_moe {
        format!("{} (={}x{}x{})", worker_gpus, row.tp, row.pp, row.dp)
    } else {
        format!("{} (={}x{}", worker_gpus, row.tp, row.pp)
    }
}

fn agg_parallel(row: &AggResult, is_moe: bool) -> String {
    if is_moe {
        format!(
            "tp{}pp{}dp{}etp{}ep{}",
            row.tp, row.pp, row.dp, row.moe_tp, row.moe_ep
        )
    } else {
        format!("tp{}pp{}", row.tp, row.pp)
    }
}

fn disagg_gpus_worker(tp: u32, pp: u32, dp: u32, is_moe: bool) -> String {
    let worker_gpus = tp * pp * dp;
    if is_moe {
        format!("{worker_gpus} (={tp}x{pp}x{dp})")
    } else {
        format!("{worker_gpus} (={tp}x{pp})")
    }
}

fn disagg_parallel(tp: u32, pp: u32, dp: u32, moe_tp: u32, moe_ep: u32, is_moe: bool) -> String {
    if is_moe {
        format!("tp{tp}pp{pp}dp{dp}etp{moe_tp}ep{moe_ep}")
    } else {
        format!("tp{tp}pp{pp}")
    }
}

fn render_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let widths = headers
        .iter()
        .enumerate()
        .map(|(idx, header)| {
            rows.iter()
                .filter_map(|row| row.get(idx))
                .map(|value| value.len())
                .fold(header.len(), usize::max)
        })
        .collect::<Vec<_>>();
    let border = table_border(&widths);
    let mut out = String::new();
    out.push_str(&border);
    out.push('\n');
    out.push_str(&table_row(
        &headers
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>(),
        &widths,
    ));
    out.push('\n');
    out.push_str(&border);
    for row in rows {
        out.push('\n');
        out.push_str(&table_row(row, &widths));
    }
    out.push('\n');
    out.push_str(&border);
    out
}

fn table_border(widths: &[usize]) -> String {
    let mut out = String::new();
    out.push('+');
    for width in widths {
        out.push_str(&"-".repeat(width + 2));
        out.push('+');
    }
    out
}

fn table_row(values: &[String], widths: &[usize]) -> String {
    let mut out = String::new();
    out.push('|');
    for (idx, width) in widths.iter().enumerate() {
        let value = values.get(idx).map(String::as_str).unwrap_or("");
        out.push(' ');
        out.push_str(&center(value, *width));
        out.push(' ');
        out.push('|');
    }
    out
}

fn center(value: &str, width: usize) -> String {
    let len = value.len();
    if len >= width {
        return value.to_string();
    }
    let pad = width - len;
    format!(
        "{}{}{}",
        " ".repeat(pad / 2),
        value,
        " ".repeat(pad - pad / 2)
    )
}

fn format_count(value: f64) -> String {
    if (value.round() - value).abs() < 1e-6 {
        format!("{:.0}", value)
    } else {
        format!("{:.2}", value)
    }
}

fn format_number(value: f64, decimals: usize) -> String {
    let raw = format!("{value:.decimals$}");
    let (int_part, frac_part) = raw.split_once('.').unwrap_or((raw.as_str(), ""));
    let mut grouped = String::new();
    for (idx, ch) in int_part.chars().rev().enumerate() {
        if idx > 0 && idx % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(ch);
    }
    let int_grouped = grouped.chars().rev().collect::<String>();
    if decimals == 0 {
        int_grouped
    } else {
        format!("{int_grouped}.{frac_part}")
    }
}

fn cmd_query_gemm(args: QueryGemmArgs) -> Result<()> {
    let version = match args.version {
        Some(version) => version,
        None => latest_database_version(default_systems_root(), &args.system, args.backend)?
            .with_context(|| format!("no database for {}/{}", args.system, args.backend))?,
    };
    let db = PerfDatabase::load(default_systems_root(), &args.system, args.backend, &version)?;
    let result = db.query_gemm(args.m, args.n, args.k, args.quant, args.database_mode)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn cmd_query_context_attention(args: QueryContextAttentionArgs) -> Result<()> {
    let version = match args.version {
        Some(version) => version,
        None => latest_database_version(default_systems_root(), &args.system, args.backend)?
            .with_context(|| format!("no database for {}/{}", args.system, args.backend))?,
    };
    let db = PerfDatabase::load(default_systems_root(), &args.system, args.backend, &version)?;
    let result = db.query_context_attention(
        args.batch_size,
        args.seq_len,
        args.prefix,
        args.num_heads,
        args.num_kv_heads,
        args.kv_cache_quant,
        args.fmha_quant,
        args.database_mode,
        args.window_size,
        args.head_size,
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn cmd_query_generation_attention(args: QueryGenerationAttentionArgs) -> Result<()> {
    let version = match args.version {
        Some(version) => version,
        None => latest_database_version(default_systems_root(), &args.system, args.backend)?
            .with_context(|| format!("no database for {}/{}", args.system, args.backend))?,
    };
    let db = PerfDatabase::load(default_systems_root(), &args.system, args.backend, &version)?;
    let result = db.query_generation_attention(
        args.batch_size,
        args.seq_len,
        args.num_heads,
        args.num_kv_heads,
        args.kv_cache_quant,
        args.database_mode,
        args.window_size,
        args.head_size,
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn cmd_query_moe(args: QueryMoeArgs) -> Result<()> {
    let version = match args.version {
        Some(version) => version,
        None => latest_database_version(default_systems_root(), &args.system, args.backend)?
            .with_context(|| format!("no database for {}/{}", args.system, args.backend))?,
    };
    let db = PerfDatabase::load(default_systems_root(), &args.system, args.backend, &version)?;
    let result = db.query_moe(
        args.num_tokens,
        args.hidden_size,
        args.inter_size,
        args.topk,
        args.num_experts,
        args.moe_tp_size,
        args.moe_ep_size,
        args.quant,
        &args.workload_distribution,
        args.is_context,
        args.moe_backend.as_deref(),
        args.database_mode,
        args.is_gated,
        args.enable_eplb,
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn cmd_model_info(args: ModelInfoArgs) -> Result<()> {
    let info = load_model_info(&args.model_path)?;
    println!("{}", serde_json::to_string_pretty(&info)?);
    Ok(())
}

fn generate_naive(
    model_path: &str,
    total_gpus: u32,
    system: &str,
    backend: BackendName,
) -> Result<GenerateResult> {
    let spec = SystemSpec::load(default_systems_root(), system)?;
    let weight_bytes = estimate_model_weight_bytes(model_path)?;
    let min_tp = calculate_min_tp(
        weight_bytes,
        spec.gpu.mem_capacity as u128,
        spec.node.num_gpus_per_node,
        total_gpus,
    )?;
    let replicas = total_gpus / min_tp;
    Ok(GenerateResult {
        model: model_path.to_string(),
        system: system.to_string(),
        backend: backend.to_string(),
        total_gpus,
        gpus_used: replicas * min_tp,
        tensor_parallel_size: min_tp,
        pipeline_parallel_size: 1,
        replicas,
        estimated_model_weight_bytes: weight_bytes,
    })
}

fn calculate_min_tp(
    model_weight_bytes: u128,
    vram_per_gpu: u128,
    gpus_per_node: u32,
    total_gpus: u32,
) -> Result<u32> {
    let required_bytes = model_weight_bytes * 3 / 2;
    let mut tp = 1_u32;
    while tp <= total_gpus {
        if vram_per_gpu * tp as u128 > required_bytes && total_gpus % tp == 0 && tp <= gpus_per_node
        {
            return Ok(tp);
        }
        tp *= 2;
    }
    bail!("model does not fit in available GPUs with naive TP rule")
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "YES"
    } else {
        "NO"
    }
}
