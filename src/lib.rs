pub mod graph;
pub mod primefield;
mod encode;
pub mod solve;
pub mod expander;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Config {
    distance_threshold: usize,
    //rs_rate: u32,
    alpha: f64,
    //beta: f64,
    r: f64,
    cn: usize,
    dn: usize,
    target_distance: f64,
}

impl Default for Config {
    fn default() -> Self {    
        const TARGET_DISTANCE: f64 = 0.07;
        const DISTANCE_THRESHOLD: usize = (1.0 / TARGET_DISTANCE) as usize - 1;
        //const RS_RATE: u32 = 2;
        const ALPHA: f64 = 0.238;
        //const BETA: f64 = 0.1205;
        const R: f64 = 1.72;
        const CN: usize = 10;
        const DN: usize = 20;
        //const COLUMN_SIZE: u32 = 128;
        Self {
            distance_threshold: DISTANCE_THRESHOLD,
            dn: DN,
            cn: CN,
            alpha: ALPHA,
            r: R,
            target_distance: TARGET_DISTANCE
        }
    }
}

impl Config {
    pub fn distance_threshold(&self) -> usize {
        self.distance_threshold
    }

    pub fn dn(&self) -> usize {
        self.dn
    }

    pub fn cn(&self) -> usize {
        self.cn
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn r(&self) -> f64 {
        self.r
    }

    pub fn target_distance(&self) -> f64 {
        self.target_distance
    }
}
