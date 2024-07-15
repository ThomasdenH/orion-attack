use nalgebra::DVector;
use orion_rust::Config;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::error::Error;
use std::iter;

use ark_ff::UniformRand;
use num_traits::{Zero, One};
use orion_rust::primefield::Fp;
use orion_rust::solve::solve_ax_is_b;

use orion_rust::expander::Expander;

const COLUMNS_TO_OPEN: usize = 1568;
const N: usize = 1 << 11;

fn main() -> Result<(), Box<dyn Error>> {
    let iterations = 100;
    let mut openings = 0;
    for i in 0..iterations {
        let config = Config::default();
        let mut rng = &mut SmallRng::from_entropy();
        let expander = Expander::new(N, rng, config);

        println!("Generated expander!");

        println!(
            "Expected minimum distance: {}/{} (δ = {})",
            (config.target_distance() * N as f64) as u64,
            N,
            config.target_distance()
        );

        // The honest y_1
        let input = DVector::from_iterator(N, iter::from_fn(|| Some(Fp::rand(&mut rng))).take(N));
        // The honest Ec(y_1). This is what we want to match in our search
        let output = expander.encode_to_slice(input.as_view());

        println!("Encoded real output.");

        let desired_evaluation = Fp::rand(&mut rng);
        let evaluation_point = Fp::rand(&mut rng);
        let evaluation_vector = DVector::from_iterator(
            N,
            iter::successors(Some(Fp::one()), |prev| Some(*prev * evaluation_point)).take(N),
        );
        //let evaluation_vector =
         //   DVector::from_iterator(N, iter::from_fn(|| Some(Fp::rand(&mut rng))).take(N));

        println!("Prepared evaluation.");

        // The indices to check equality at
        let random_indices = expander.random_output_indices(COLUMNS_TO_OPEN, &mut rng);

        println!("Generated opening set.");

        let matrix = expander.matrix_checked_indices_from_input(&random_indices);
        let matrix_output = (&matrix) * (&input);
        assert_eq!(matrix_output.len(), COLUMNS_TO_OPEN);
        for (matrix_mul_output, index) in matrix_output.iter().zip(random_indices.iter()) {
            assert_eq!(*matrix_mul_output, output[*index]);
        }

        // Append a new row for the evaluation constraint
        let mut matrix = matrix.insert_row(0, Fp::zero());
        for column in 0..N {
            matrix[(0, column)] = evaluation_vector[column];
        }
        let matrix_output = matrix_output.insert_row(0, desired_evaluation);

        // Solve the system
        let forged_message = match solve_ax_is_b(matrix.clone(), matrix_output.clone()) {
            Ok(m) => m,
            Err(e) => {
                println!("System could not be solved: {:?}", e);

                println!("{}", &matrix);
                println!("{}", &matrix_output);
                println!("{}", &input);

                assert_eq!(matrix * input, matrix_output);

                return Ok(());
            }
        };

        // Check the message. First encode,
        let encoding = expander.encode_to_slice(forged_message.as_view());
        // ... then check all openings, ...
        for &index in random_indices.iter() {
            assert_eq!(encoding[index], output[index]);
        }
        // ... and finally check the evaluation!
        //assert_eq!(evaluation_vector.dot(&forged_message), desired_evaluation);

        openings += 1;
        println!("Found opening!");
        println!(
            "Opened {} of {} ({}%)",
            openings,
            i + 1,
            100.0 * (openings as f64) / (i + 1) as f64
        );
    }

    Ok(())
}
