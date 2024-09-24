use std::ops::Mul;

use nalgebra::{DVector, Dim, VectorView};
use rand::Rng;

use crate::primefield::Fp;
use ark_ff::UniformRand;

/// A graph representing a random expander. The graph can be used as if it is a
/// matrix.
#[derive(Debug, Default)]
pub struct Graph {
    pub neighbor: Vec<Vec<(usize, Fp)>>,
    r_neighbor: Vec<Vec<(usize, Fp)>>,
}

impl Graph {
    /// Create a new random expander.
    ///
    /// - `l`: The input size.
    /// - `r`: The output size.
    /// - `degree`: The degree of each input node.
    /// - `rng`: The random number generator to use.
    ///
    /// ```
    /// use orion_rust::graph::Graph;
    /// use rand::thread_rng;
    ///
    /// let graph = Graph::generate_random_expander(100, 200, 10, &mut thread_rng());
    /// assert_eq!(graph.l(), 100);
    /// assert_eq!(graph.r(), 200);
    /// assert_eq!(graph.degree(), 10);
    /// ```
    pub fn generate_random_expander(
        l: usize,
        r: usize,
        degree: usize,
        rng: &mut impl Rng,
    ) -> Graph {
        let mut neighbor = vec![Vec::new(); l];
        let mut r_neighbor = vec![Vec::new(); r];
        for (i, neighbor) in neighbor.iter_mut().enumerate() {
            *neighbor = (0..degree)
                .map(|_| {
                    let target: usize = rng.gen_range(0..r);
                    let weight: Fp = Fp::rand(rng);
                    (target, weight)
                })
                .collect();
            for (target, weight) in neighbor {
                r_neighbor[*target].push((i, *weight));
            }
        }

        let graph = Graph {
            neighbor,
            r_neighbor,
        };
        assert_eq!(graph.l(), l);
        assert_eq!(graph.r(), r);
        assert_eq!(graph.degree(), degree);
        graph
    }

    /// The amount of inputs for this graph.
    pub fn l(&self) -> usize {
        self.neighbor.len()
    }

    /// The amount of outputs for this graph.
    pub fn r(&self) -> usize {
        self.r_neighbor.len()
    }

    /// The out-degree of the inputs for this graph.
    pub fn degree(&self) -> usize {
        self.neighbor[0].len()
    }
}

impl<R: Dim> Mul<VectorView<'_, Fp, R>> for &Graph {
    type Output = DVector<Fp>;
    fn mul(self, rhs: VectorView<'_, Fp, R>) -> Self::Output {
        assert_eq!(rhs.nrows(), self.l());
        let mut output = DVector::zeros(self.r());
        for (from_index, neighbors) in self.neighbor.iter().enumerate() {
            for (to_index, weight) in neighbors.iter().copied() {
                output[to_index] += weight * rhs[from_index];
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use nalgebra::{DVector, Dyn};
    use rand::{rngs::SmallRng, SeedableRng};
    use ark_ff::UniformRand;

    use crate::primefield::Fp;

    use super::Graph;

    #[test]
    fn test_nalgebra_finishes() {
        let _ = DVector::from_iterator(70, iter::from_fn(|| Some(1)));
    }

    #[test]
    fn test_mul() {
        let mut rand = SmallRng::seed_from_u64(226453645362346);
        let vector = DVector::from_iterator(70, iter::from_fn(|| Some(Fp::rand(&mut rand))).take(70));
        let graph = Graph::generate_random_expander(70, 10, 5, &mut rand);
        let result = &graph * vector.as_view::<Dyn, _, _, _>();
        assert_eq!(result.nrows(), 10);
        assert_eq!(result.ncols(), 1);
    }
}
