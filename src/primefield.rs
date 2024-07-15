use ark_ff::{Fp64, MontBackend};

#[derive(ark_ff::MontConfig)]
#[modulus = "2305843009213693951"]
#[generator = "7"]
pub struct F127Config;
pub type Fp = Fp64<MontBackend<F127Config, 1>>;
