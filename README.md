# Orion attack implementation
This is a demonstration of an attack on the Orion proof system. It is implemented in Rust and can be executed by running
```bash
cargo run --release
```
All dependencies are included in `Cargo.toml`, requiring no additional setup.

Running the associated tests can be done using `cargo test`.

The attack is explained in [`A Crack in the Firmament: Restoring Soundness of the Orion Proof System and More'](https://eprint.iacr.org/2024/1164), Section 3. Comments are provided to clarify the steps taken in the program. The code will repeatedly generate a random system and find a codeword to forge a proof, keeping track of the success rate (the code is never expected to fail).

## Code organisation
The organisation of the code is relatively self-explanatory:
- `main.rs` contains the attack code.
- `lib.rs` contains configuration options for the attack. The default options are those used in Orion.
- `primefield.rs` contains the finite field used.
- `graph.rs` implements an expander graph.
- `spielman.rs` implements a Spielman code, utilizing the graphs defined in `graph.rs`.
- `solve.rs` implements a linear algebra solver.
