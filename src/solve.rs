use crate::primefield::Fp;
use ark_ff::Field;
use nalgebra::allocator::Allocator;
use nalgebra::constraint::{AreMultipliable, ShapeConstraint};
use nalgebra::{
    DVector, DefaultAllocator, Dim, Dyn, Matrix, RawStorageMut, Storage, StorageMut, Vector, U1,
};
use num_traits::{One, Zero};
use std::fmt::Debug;
use std::ops::MulAssign;
use thiserror::Error;

#[derive(Eq, PartialEq, Clone, Copy, Debug, Hash, Error)]
#[error("there is no solution")]
pub struct NoSolution;

pub fn solve_ax_is_b<
    R: Dim,
    C: Dim,
    R1: Dim,
    Storage1: StorageMut<Fp, R, C> + Storage<Fp, R, C> + Debug,
    Storage2: StorageMut<Fp, R1> + Storage<Fp, R1> + Debug,
>(
    mut a: Matrix<Fp, R, C, Storage1>,
    mut b: Vector<Fp, R1, Storage2>,
) -> Result<DVector<Fp>, NoSolution>
where
    DefaultAllocator: Allocator<Fp, U1, U1>,
    ShapeConstraint: AreMultipliable<U1, C, Dyn, U1>,
    DefaultAllocator: Allocator<Fp, R, U1>,
    ShapeConstraint: AreMultipliable<R, C, Dyn, U1>,
{
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(b.ncols(), 1);
    row_reduce(&mut a, &mut b);
    // Do a simple check for solvability
    for (row_a, b) in a.row_iter().zip(b.row_iter()) {
        // Any zero row of a must correspond to a zero entry in b.
        if row_a.iter().all(|a| a.is_zero()) && b.iter().any(|a| !a.is_zero()) {
            return Err(NoSolution);
        }
    }

    let mut solution = DVector::zeros(a.ncols());
    for row_index in (0..a.nrows()).rev() {
        let mut row_a = a.row_mut(row_index);
        let sol_b = b.row(row_index);
        let current_solution = (&row_a) * (&solution) - sol_b;
        // We want the current solution to be equal to b.
        if let Some((pos, pivot)) = row_a
            .iter_mut()
            .enumerate()
            .find(|(_pos, el)| !el.is_zero()) {
                // If this pivot exist, we can adjust the solution. Otherwise,
                // the entire row must be zero and therefore `b` is zero as well.
                assert!(pivot.is_one());
                solution[pos] -= current_solution[0];
            }
            assert_eq!(row_a * (&solution), sol_b);
    }
    assert_eq!(a * (&solution), b);
    Ok(solution)
}

fn row_reduce<
    R: Dim,
    C: Dim,
    R1: Dim,
    Storage1: RawStorageMut<Fp, R, C>,
    Storage2: RawStorageMut<Fp, R1>,
>(
    a: &mut Matrix<Fp, R, C, Storage1>,
    b: &mut Vector<Fp, R1, Storage2>,
) {
    debug_assert_eq!(a.nrows(), a.nrows());
    // Reduce to row echelon form
    let mut column = 0;
    // Loop invariant: the matrix is in row echelon form for any row before
    // `row_pivot`.
    for pivot_row in 0..a.nrows() {
        /*let promilage = 1000 * pivot_row / a.nrows();
        println!(
            "Pivot row: {pivot_row} / {} ({}.{}%)",
            a.nrows(),
            promilage / 10,
            promilage % 10
        );*/
        // We want to find a row with a non-zero pivot
        debug_assert!(column >= pivot_row);
        // Find a row with a non-zero pivot. Go through all columns in case this
        // column is already entirely zero.
        let (row_index_with_non_zero_pivot, new_column) =
            match a.column_iter().enumerate().skip(column).find_map(
                |(pivot_column, column)| -> Option<(usize, usize)> {
                    // Try to find a row with a non-zero pivot at this column
                    column
                        .iter()
                        .enumerate()
                        .skip(pivot_row)
                        .find(|(_pos, value)| !value.is_zero())
                        .map(|(row_index, _)| (row_index, pivot_column))
                },
            ) {
                // This returns None if all remaining rows are entirely zero. In
                // this case, the matrix is already in row-echelon form so we can
                // just return.
                None => return,
                Some(a) => a,
            };
        // We now know that some more columns are entirely zero.
        column = new_column;
        // Switch row with target row.
        a.swap_rows(pivot_row, row_index_with_non_zero_pivot);
        b.swap_rows(pivot_row, row_index_with_non_zero_pivot);
        // The pivot is now at (pivot_row, column)
        debug_assert!(!a[(pivot_row, column)].is_zero());
        // Reduce the pivot to 1
        let inverse: Fp = a[(pivot_row, column)]
            .inverse()
            .expect("every non-zero element should have an inverse");
        // We skip the first `column` columns since they must be zero already
        let total_columns = a.ncols();
        debug_assert!(a
            .row(pivot_row)
            .columns(0, column)
            .iter()
            .all(|item| item.is_zero()));
        a.row_mut(pivot_row)
            .columns_mut(column, total_columns - column)
            .mul_assign(inverse);
        b.row_mut(pivot_row).mul_assign(inverse);

        // Reduce all subsequent rows. Actually don't reduce rows that we have
        // already seen. It won't help for solving the system.
        for other_row in (pivot_row + 1)..a.nrows() {
            // Subtract this many times as to make this value zero.
            let times = a[(other_row, column)];

            // Here again, we can skip the first columns because they must be
            // zero
            debug_assert!(a
                .row(pivot_row)
                .columns(0, column)
                .iter()
                .all(|a| a.is_zero()));
            for j in column..a.ncols() {
                let val: Fp = a[(pivot_row, j)];
                a[(other_row, j)] -= val * times;
            }
            let val = b[pivot_row] * times;
            b[other_row] -= val;
        }
        for row in (pivot_row + 1)..a.nrows() {
            if row == pivot_row {
                debug_assert!(a[(row, column)].is_one());
            } else {
                debug_assert!(a[(row, column)].is_zero());
            }
        }
        column += 1;
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::UniformRand;
    use num_traits::{One, Zero};
    use rand::{rngs::SmallRng, SeedableRng};
    use std::error::Error;

    use nalgebra::{matrix, DMatrix, DVector, Matrix3x1, Matrix3x4};

    use crate::{
        primefield::Fp,
        solve::{row_reduce, NoSolution},
    };

    use super::solve_ax_is_b;

    #[test]
    fn simple_test() {
        let mut matrix = Matrix3x4::new(
            Fp::from(1),
            Fp::from(2),
            Fp::from(45),
            Fp::from(4),
            Fp::from(1423),
            Fp::from(2556),
            Fp::from(45),
            Fp::from(4),
            Fp::from(12),
            Fp::from(2346),
            Fp::from(4532),
            Fp::from(4),
        );
        let mut column = Matrix3x1::new(Fp::from(1), Fp::from(2), Fp::from(3));
        row_reduce(&mut matrix, &mut column);
        assert!(matrix[(0, 0)].is_one());
        assert!(matrix[(1, 0)].is_zero());
        assert!(matrix[(2, 0)].is_zero());
        assert!(matrix[(1, 1)].is_one());
        assert!(matrix[(2, 1)].is_zero());
        assert!(matrix[(2, 2)].is_one());
    }

    #[test]
    fn another_test() -> Result<(), Box<dyn Error>> {
        let matrix = Matrix3x4::new(
            Fp::from(1),
            Fp::from(2),
            Fp::from(45),
            Fp::from(4),
            Fp::from(1423),
            Fp::from(2556),
            Fp::from(45),
            Fp::from(4),
            Fp::from(12),
            Fp::from(2346),
            Fp::from(4532),
            Fp::from(4),
        );
        let column = Matrix3x1::new(Fp::from(1), Fp::from(2), Fp::from(3));
        let solution = solve_ax_is_b(matrix.clone(), column.clone())?;
        assert_eq!(matrix * solution, column);
        Ok(())
    }

    #[test]
    fn test_random_matrices() -> Result<(), Box<dyn Error>> {
        let mut rng = SmallRng::from_entropy();
        for _ in 0..100 {
            let matrix = DMatrix::from_fn(100, 1000, |_, _| Fp::rand(&mut rng));
            let x = DVector::from_fn(1000, |_, _| Fp::rand(&mut rng));
            let b = (&matrix) * x;
            assert_eq!(matrix.clone() * solve_ax_is_b(matrix, b.clone())?, b);
        }
        Ok(())
    }

    #[test]
    fn test_unsolvable() {
        let matrix = matrix! {
            Fp::from(3),  Fp::from(4), Fp::from(5);
            Fp::from(3),  Fp::from(4), Fp::from(5);
        };
        let solution = matrix! {
            Fp::from(3);
            Fp::from(5)
        };
        assert_eq!(solve_ax_is_b(matrix, solution), Err(NoSolution));
    }
}
