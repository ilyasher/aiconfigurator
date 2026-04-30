use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Div, Mul};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuantMapping {
    pub memory: f64,
    pub compute: f64,
    pub name: &'static str,
}

macro_rules! quant_enum {
    ($name:ident { $($variant:ident => ($memory:expr, $compute:expr, $label:expr)),+ $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[allow(non_camel_case_types)]
        pub enum $name {
            $($variant),+
        }

        impl $name {
            pub fn mapping(self) -> QuantMapping {
                match self {
                    $(Self::$variant => QuantMapping { memory: $memory, compute: $compute, name: $label }),+
                }
            }

            pub fn as_str(self) -> &'static str {
                self.mapping().name
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl FromStr for $name {
            type Err = anyhow::Error;

            fn from_str(s: &str) -> Result<Self> {
                match s {
                    $($label => Ok(Self::$variant),)+
                    other => bail!("unsupported {} value: {other}", stringify!($name)),
                }
            }
        }
    };
}

quant_enum!(GemmQuantMode {
    bfloat16 => (2.0, 1.0, "bfloat16"),
    int8_wo => (1.0, 1.0, "int8_wo"),
    int4_wo => (0.5, 1.0, "int4_wo"),
    fp8 => (1.0, 2.0, "fp8"),
    fp8_static => (1.0, 2.0, "fp8_static"),
    sq => (1.0, 2.0, "sq"),
    fp8_block => (1.0, 2.0, "fp8_block"),
    fp8_ootb => (1.0, 2.0, "fp8_ootb"),
    nvfp4 => (9.0 / 16.0, 4.0, "nvfp4"),
});

quant_enum!(MoeQuantMode {
    bfloat16 => (2.0, 1.0, "bfloat16"),
    fp8 => (1.0, 2.0, "fp8"),
    int4_wo => (0.5, 1.0, "int4_wo"),
    fp8_block => (1.0, 2.0, "fp8_block"),
    w4afp8 => (0.5, 2.0, "w4afp8"),
    nvfp4 => (9.0 / 16.0, 4.0, "nvfp4"),
    w4a16_mxfp4 => (0.5, 1.0, "w4a16_mxfp4"),
    w4a8_mxfp4_mxfp8 => (0.5, 2.0, "w4a8_mxfp4_mxfp8"),
});

quant_enum!(FmhaQuantMode {
    bfloat16 => (2.0, 1.0, "bfloat16"),
    fp8 => (1.0, 2.0, "fp8"),
    fp8_block => (1.0, 2.0, "fp8_block"),
});

quant_enum!(KvCacheQuantMode {
    bfloat16 => (2.0, 0.0, "bfloat16"),
    int8 => (1.0, 0.0, "int8"),
    fp8 => (1.0, 0.0, "fp8"),
});

quant_enum!(CommQuantMode {
    half => (2.0, 0.0, "half"),
    int8 => (1.0, 0.0, "int8"),
    fp8 => (1.0, 0.0, "fp8"),
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum BackendName {
    trtllm,
    sglang,
    vllm,
}

impl fmt::Display for BackendName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::trtllm => "trtllm",
            Self::sglang => "sglang",
            Self::vllm => "vllm",
        })
    }
}

impl FromStr for BackendName {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "trtllm" => Ok(Self::trtllm),
            "sglang" => Ok(Self::sglang),
            "vllm" => Ok(Self::vllm),
            other => bail!("unsupported backend: {other}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum DatabaseMode {
    SILICON,
    HYBRID,
    EMPIRICAL,
    SOL,
    SOL_FULL,
}

impl FromStr for DatabaseMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "SILICON" => Ok(Self::SILICON),
            "HYBRID" => Ok(Self::HYBRID),
            "EMPIRICAL" => Ok(Self::EMPIRICAL),
            "SOL" => Ok(Self::SOL),
            "SOL_FULL" => Ok(Self::SOL_FULL),
            other => bail!("unsupported database mode: {other}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PerformanceResult {
    pub latency: f64,
    pub energy: f64,
}

impl PerformanceResult {
    pub fn new(latency: f64, energy: f64) -> Self {
        Self { latency, energy }
    }

    pub fn power(self) -> f64 {
        if self.latency > 1e-9 {
            self.energy / self.latency
        } else {
            0.0
        }
    }
}

impl Add for PerformanceResult {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.latency + rhs.latency, self.energy + rhs.energy)
    }
}

impl Mul<f64> for PerformanceResult {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.latency * rhs, self.energy * rhs)
    }
}

impl Div<f64> for PerformanceResult {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self::new(self.latency / rhs, self.energy / rhs)
    }
}
