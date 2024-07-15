use std::collections::HashSet;

use nalgebra::{DMatrix, DVector, Dyn, VectorView, VectorViewMut};
use num_traits::{One, Zero};
use rand::Rng;

use crate::{graph::Graph, primefield::Fp, Config};

#[derive(Debug)]
pub struct Expander {
    n: usize,
    c: Vec<Graph>,
    d: Vec<Graph>,
    config: Config,
}

impl Expander {
    pub fn output_size(&self) -> usize {
        if self.n < self.config.distance_threshold() {
            self.n
        } else {
            assert!(!self.c.is_empty());
            assert!(!self.d.is_empty());
            self.c[0].l() + self.d[0].l() + self.d[0].r()
        }
    }

    fn input_size(&self) -> usize {
        self.n
    }

    fn init_recursive(
        expander: &mut Expander,
        n: usize,
        depth: usize,
        rng: &mut impl Rng,
    ) -> usize {
        if n <= expander.config.distance_threshold() {
            n
        } else {
            assert_eq!(expander.c.len(), depth);
            let alpha_n = (expander.config.alpha() * (n as f64)) as usize;
            expander.c.push(Graph::generate_random_expander(
                n,
                alpha_n,
                expander.config.cn(),
                rng,
            ));
            // Already push a graph to make sure it exists later
            expander.d.push(Graph::default());

            let l = Expander::init_recursive(expander, alpha_n, depth + 1, rng);
            let n_r1_l = (n as f64 * (expander.config.r() - 1.0) - l as f64) as usize;
            expander.d[depth] =
                Graph::generate_random_expander(l, n_r1_l, expander.config.dn(), rng);
            n + l + n_r1_l
        }
    }

    pub fn new(n: usize, rng: &mut impl Rng, config: Config) -> Expander {
        let mut expander = Expander {
            n,
            c: Vec::new(),
            d: Vec::new(),
            config,
        };
        Expander::init_recursive(&mut expander, n, 0, rng);
        assert_eq!(expander.c.len(), expander.d.len());
        expander
    }

    fn encode_to_slice_recursive(
        &self,
        input: VectorView<'_, Fp, Dyn>,
        mut output: VectorViewMut<'_, Fp, Dyn>,
        depth: usize,
    ) {
        if input.len() <= self.config.distance_threshold() {
            assert_eq!(input.nrows(), output.nrows());
            // Simply copy the output
            output.copy_from(&input);
        } else {
            // First copy the input
            for (to, from) in output.iter_mut().zip(input.iter()) {
                *to = *from;
            }

            // Multiply and encode
            let encoded = &self.c[depth] * input;
            let output_2_length = self.d[depth].l();
            let mut second_part_slice =
                output.rows_range_mut(input.len()..input.len() + output_2_length);
            self.encode_to_slice_recursive(
                encoded.as_view(),
                second_part_slice.as_view_mut(),
                depth + 1,
            );

            // Multiply just encoded part again
            let multiplied_again = &self.d[depth] * second_part_slice.as_view::<Dyn, _, _, _>();

            output
                .rows_mut(input.len() + output_2_length, multiplied_again.len())
                .copy_from(&multiplied_again);
        }
    }

    pub fn encode_to_slice(&self, input: VectorView<Fp, Dyn>) -> DVector<Fp>
where {
        assert_eq!(input.len(), self.input_size());
        let mut vec = DVector::zeros(self.output_size());
        self.encode_to_slice_recursive(input.as_view(), vec.as_view_mut(), 0);
        vec
    }

    /// Get the matrix representing the selection of certain indices of the
    /// output.
    ///
    /// ```text
    ///  _________
    /// |         |
    ///  \        |
    ///   \       |
    ///    \______|
    /// | selected indices
    ///           | code output
    /// ```
    fn matrix_checked_indices_from_output(&self, indices: &[usize]) -> DMatrix<Fp> {
        let mut matrix = DMatrix::from_element(indices.len(), self.output_size(), Fp::zero());
        for (going_to, &output_index) in indices.iter().enumerate() {
            matrix[(going_to, output_index)] = Fp::one();
        }
        matrix
    }

    /// We work backwards to build a matrix for this code. To keep the matrix
    /// small enough, we work backwards from the selected indices, reducing the
    /// width of the matrix each time.
    pub fn matrix_checked_indices_from_input(&self, indices: &[usize]) -> DMatrix<Fp> {
        let mut matrix = self.matrix_checked_indices_from_output(indices);
        self.matrix_encode_recursively(&mut matrix, 0);
        assert_eq!(matrix.ncols(), self.input_size());
        assert_eq!(matrix.nrows(), indices.len());
        matrix
    }

    /// Adapt the given matrix such that we add one layer of encoding to the
    /// input. Where the matrix currently maps (m || encoding) to (output indices),
    /// the resulting matrix will map (m) to (output indices).
    ///
    /// Note that, because of recursion, there may be more columns at the stat
    /// of the matrix. It is always true that the encoding happens at the last
    /// columns, and is determined by depth.
    fn matrix_encode_recursively(&self, matrix: &mut DMatrix<Fp>, depth: usize) {
        if self.c.len() <= depth {
            // Encoding is simply the identity matrix
        } else {
            // Reverse the last matrix encoding.
            //  ___ ___
            // |   |___|    Direct copy
            //  \  |___|    Result of encoding
            //   \_|_/   <- Reduce column set by multiplying with D
            //

            // These indices are the same before and after multiplication by D,
            // since parts 1 & 2 are just copied.
            println!(
                "Multiplication by D. Depth: {depth}. {} x {} = {}",
                matrix.nrows(),
                matrix.ncols(),
                matrix.len()
            );
            let part_3_start = matrix.ncols() - self.d[depth].r();
            let part_2_start = part_3_start - self.d[depth].l();
            for (source_index, neighbors) in self.d[depth].neighbor.iter().enumerate() {
                for (dest_index, weight) in neighbors.iter().copied() {
                    let source_index_in_matrix = part_2_start + source_index;
                    let dest_index_in_matrix = part_3_start + dest_index;
                    // We multiply the value at `source_index_in_matrix` by `weight`
                    // and put the result at `dest_index_in_matrix`. Here this means
                    // multiplying the column at `dest_index_in_matrix` by `weight`
                    // and adding it to `source_index_in_matrix`.
                    for row in 0..matrix.nrows() {
                        let val = matrix[(row, dest_index_in_matrix)] * weight;
                        matrix[(row, source_index_in_matrix)] += val;
                    }
                }
            }
            // Remove this part
            *matrix = matrix
                .clone()
                .remove_columns(part_3_start, self.d[depth].r());

            // Part 2 is recursive encoding.
            //
            // Result:
            //  ___ ___
            // |   |___| Direct copy
            //  \__|___| Multiplication by C
            println!(
                "Encoding. Depth: {depth}. {} x {} = {}",
                matrix.nrows(),
                matrix.ncols(),
                matrix.len()
            );
            self.matrix_encode_recursively(matrix, depth + 1);

            // Lastly, remove the first matrix multiplication by C.
            // ```text
            //  ___                       _
            // |   | Direct copy         |_| Input message
            //  \ _| Multiplication by C |_/
            // ```
            println!(
                "Multiplication by C. Depth: {depth}. {} x {} = {}",
                matrix.nrows(),
                matrix.ncols(),
                matrix.len()
            );
            let dest_start = matrix.ncols() - self.c[depth].r();
            let source_start = dest_start - self.c[depth].l();
            for (source_index, neighbors) in self.c[depth].neighbor.iter().enumerate() {
                for (dest_index, weight) in neighbors.iter().copied() {
                    let source_index_in_matrix = source_start + source_index;
                    let dest_index_in_matrix = dest_start + dest_index;
                    for row in 0..matrix.nrows() {
                        let val = matrix[(row, dest_index_in_matrix)] * weight;
                        matrix[(row, source_index_in_matrix)] += val;
                    }
                }
            }
            *matrix = matrix.clone().remove_columns(dest_start, self.c[depth].r());
        }
    }

    /// Get a vector with random indices to open the output of this expander at.
    pub fn random_output_indices(&self, count: usize, rng: &mut impl Rng) -> Vec<usize> {
            let mut set = HashSet::new();
            while set.len() < count {
                set.insert(rng.gen_range(0..self.output_size()));
            }
            set.into_iter().collect()
        
    }
}
