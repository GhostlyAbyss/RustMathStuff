use std::cmp::Ordering;
use std::collections::HashMap;

use polars::prelude::*;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use crate::errors::split_error::SplitError;

pub struct Split;

impl Split {
    pub fn train_test_split(
        x: &DataFrame,
        y: &Column,
        test_size: Option<f32>,
        train_size: Option<f32>,
        random_state: Option<u64>,
        stratify: Option<&Series>,
        shuffle: Option<bool>,
    ) -> Result<(DataFrame, DataFrame, Column, Column), SplitError> {
        let n = x.height();

        if y.len() != n {
            return Err(invalid_input("X and y must have the same number of rows"));
        }

        if let Some(s) = stratify {
            if s.len() != n {
                return Err(invalid_input("stratify must have the same number of rows as X"));
            }
        }

        let (test_size, train_size) = match (test_size, train_size) {
            (Some(test), Some(train)) => (test, train),
            (Some(test), None) => (test, 1.0 - test),
            (None, Some(train)) => (1.0 - train, train),
            (None, None) => (0.25, 0.75),
        };

        if !(0.0..=1.0).contains(&test_size) || !(0.0..=1.0).contains(&train_size) {
            return Err(invalid_input("test_size and train_size must be between 0.0 and 1.0"));
        }

        let seed = random_state.unwrap_or(42);
        let shuffle = shuffle.unwrap_or(true);

        let test_n = ((n as f32) * test_size).round() as usize;
        let test_n = test_n.min(n);
        let train_n = n - test_n;

        let (train_idx, test_idx) = match stratify {
            Some(labels) => {
                if !shuffle {
                    return Err(invalid_input("stratify requires shuffle = true"));
                }
                Self::stratified_indices(labels, test_n, seed)?
            }
            None => {
                let mut idx: Vec<usize> = (0..n).collect();

                if shuffle {
                    let mut rng = StdRng::seed_from_u64(seed);
                    idx.shuffle(&mut rng);
                }

                let train_idx = idx[..train_n].to_vec();
                let test_idx = idx[train_n..].to_vec();
                (train_idx, test_idx)
            }
        };

        let x_train = Self::take_df(x, &train_idx)?;
        let x_test = Self::take_df(x, &test_idx)?;
        let y_train = Self::take_col(y, &train_idx)?;
        let y_test = Self::take_col(y, &test_idx)?;

        Ok((x_train, x_test, y_train, y_test))
    }

    fn take_df(df: &DataFrame, idx: &[usize]) -> Result<DataFrame, SplitError> {
        let idx_u32: Vec<u32> = idx.iter().map(|&i| i as u32).collect();

        let cols = df
            .columns()
            .iter()
            .map(|c| c.take_slice(&idx_u32).map_err(|e| invalid_input(e.to_string())))
            .collect::<Result<Vec<_>, _>>()?;

        let height = idx_u32.len();
        DataFrame::new(height, cols).map_err(|e| invalid_input(e.to_string()))
    }

    fn take_col(col: &Column, idx: &[usize]) -> Result<Column, SplitError> {
        let idx_u32: Vec<u32> = idx.iter().map(|&i| i as u32).collect();
        col.take_slice(&idx_u32).map_err(|e| invalid_input(e.to_string()))
    }

    fn stratified_indices(
        labels: &Series,
        test_n: usize,
        seed: u64,
    ) -> Result<(Vec<usize>, Vec<usize>), SplitError> {
        #[derive(Debug)]
        struct Group {
            idxs: Vec<usize>,
            floor_test: usize,
            frac: f32,
        }

        let n = labels.len();
        let mut buckets: HashMap<String, Vec<usize>> = HashMap::new();

        for i in 0..n {
            let key = labels
                .str_value(i)
                .map_err(|e| invalid_input(e.to_string()))?
                .into_owned();

            buckets.entry(key).or_default().push(i);
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let mut groups: Vec<Group> = Vec::with_capacity(buckets.len());

        for mut idxs in buckets.into_values() {
            idxs.shuffle(&mut rng);

            let ideal = (idxs.len() as f32) * (test_n as f32) / (n as f32);
            let floor_test = ideal.floor() as usize;
            let frac = ideal - floor_test as f32;

            groups.push(Group {
                idxs,
                floor_test,
                frac,
            });
        }

        let assigned: usize = groups.iter().map(|g| g.floor_test).sum();
        let mut remaining = test_n.saturating_sub(assigned);

        groups.sort_by(|a, b| b.frac.partial_cmp(&a.frac).unwrap_or(Ordering::Equal));

        for g in groups.iter_mut() {
            if remaining == 0 {
                break;
            }
            if g.floor_test < g.idxs.len() {
                g.floor_test += 1;
                remaining -= 1;
            }
        }

        let mut train = Vec::with_capacity(n - test_n);
        let mut test = Vec::with_capacity(test_n);

        for g in groups {
            let split_at = g.floor_test.min(g.idxs.len());
            test.extend_from_slice(&g.idxs[..split_at]);
            train.extend_from_slice(&g.idxs[split_at..]);
        }

        train.shuffle(&mut rng);
        test.shuffle(&mut rng);

        Ok((train, test))
    }
}
fn invalid_input(msg: impl Into<String>) -> SplitError {
    SplitError::InvalidInput(msg.into())
}