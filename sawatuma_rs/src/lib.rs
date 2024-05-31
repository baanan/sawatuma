use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use itertools::Itertools;
use pyo3::prelude::*;

const LISTENING_COUNTS_FILE_IN: &str = "listening_counts.tsv";
const LISTENING_COUNTS_FILE_OUT: &str = "listening_counts_filtered";

/// Filters listening counts at `{root}/listening_counts.tsv` and outputs them to
/// `{root}/listening_counts_filtered.tsv`
///
/// Returns the amount of lines written
#[pyfunction]
#[pyo3(signature = (user_divisor, track_divisor, root = "root"))]
fn filter_listening_counts(
    user_divisor: usize,
    track_divisor: usize,
    root: &str,
) -> PyResult<usize> {
    let output = &format!("{root}/{LISTENING_COUNTS_FILE_OUT}_{user_divisor}_{track_divisor}.tsv");
    let output = Path::new(output);

    if output.exists() {
        println!("  filtered output already exists, calculating the linecount");
        let reader = BufReader::new(File::open(output)?);
        return Ok(reader.lines().count() - 1);
    }

    let input = format!("{root}/{LISTENING_COUNTS_FILE_IN}");
    let input = BufReader::new(File::open(input)?);

    let mut lines = input.lines();
    let header = lines.next().expect("file must at least have a header")?;

    let mut output = BufWriter::new(File::create(output)?);

    writeln!(output, "{header}")?;

    let mut line_index = 0;

    for line in lines
        .map_while(Result::ok)
        .filter(|row| valid_row(row, user_divisor, track_divisor))
    {
        writeln!(output, "{line}")?;
        line_index += 1;
    }

    Ok(line_index)
}

/// Filters listening counts at `{root}/listening_counts.tsv` and outputs them to
/// `{root}/listening_counts_filtered.tsv`
///
/// Returns the amount of lines written
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

fn valid_row(row: &str, user_divisor: usize, track_divisor: usize) -> bool {
    let (user_id, track_id) = row
        .split('\t')
        .map(|val| val.parse::<usize>().expect("failed to parse int"))
        .next_tuple()
        .expect("failed to parse csv row");

    user_id % user_divisor == 0 && track_id % track_divisor == 0
}

/// A Python module implemented in Rust.
#[pymodule]
fn sawatuma_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cut_listening_counts, m)?)?;
    m.add_function(wrap_pyfunction!(filter_listening_counts, m)?)?;
    Ok(())
}
