use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use pyo3::prelude::*;

const LISTENING_COUNTS_FILE_IN: &str = "listening_counts.tsv";
const LISTENING_COUNTS_FILE_OUT: &str = "listening_counts_filtered";

/// Cuts the lines in the dataset by the specified `divisor`
///
/// For example, if the divisor is 2, this skips every other line.
#[pyfunction]
#[pyo3(signature = (divisor, root = "root"))]
fn cut_listening_counts(divisor: usize, root: &str) -> PyResult<()> {
    let output = &format!("{root}/{LISTENING_COUNTS_FILE_OUT}_{divisor}.tsv");
    let output = Path::new(output);

    if output.exists() {
        return Ok(());
    }

    let input = format!("{root}/{LISTENING_COUNTS_FILE_IN}");
    let input = BufReader::new(File::open(input)?);

    let mut lines = input.lines();
    let header = lines.next().expect("file must at least have a header")?;

    let mut output = BufWriter::new(File::create(output)?);

    writeln!(output, "{header}")?;

    for line in lines.map_while(Result::ok).step_by(2) {
        writeln!(output, "{line}")?;
    }

    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn sawatuma_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cut_listening_counts, m)?)?;
    Ok(())
}
