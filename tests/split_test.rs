#[cfg(test)]
mod tests {
    use polars::prelude::*;
    use custom_math_ml_stuff::rs_learn::split::Split;

    fn sample_df() -> (DataFrame, Column) {
        let df = df![
            "f1" => &[1,2,3,4,5,6,7,8],
            "f2" => &[10,20,30,40,50,60,70,80]
        ]
            .unwrap();

        let y = Series::new("target".into(), &[0,0,1,1,0,1,0,1]).into();

        (df, y)
    }

    #[test]
    fn test_basic_split_sizes() {
        let (x, y) = sample_df();

        let split = Split::new()
            .test_size(0.25)
            .random_state(42)
            .shuffle(true);

        let (x_train, x_test, y_train, y_test) =
            split.train_test_split(&x, &y).unwrap();

        assert_eq!(x_train.height(), 6);
        assert_eq!(x_test.height(), 2);

        assert_eq!(y_train.len(), 6);
        assert_eq!(y_test.len(), 2);
    }

    #[test]
    fn test_split_without_shuffle() {
        let (x, y) = sample_df();

        let split = Split::new()
            .test_size(0.25)
            .shuffle(false);

        let (x_train, x_test, _, _) =
            split.train_test_split(&x, &y).unwrap();

        assert_eq!(x_train.height(), 6);
        assert_eq!(x_test.height(), 2);

        assert_eq!(
            x_train.column("f1").unwrap().i32().unwrap().get(0),
            Some(1)
        );

        assert_eq!(
            x_test.column("f1").unwrap().i32().unwrap().get(0),
            Some(7)
        );
    }

    #[test]
    fn test_random_state_reproducibility() {
        let (x, y) = sample_df();

        let split = Split::new()
            .test_size(0.25)
            .random_state(42)
            .shuffle(true);

        let (a_train, _, _, _) =
            split.train_test_split(&x, &y).unwrap();

        let (b_train, _, _, _) =
            split.train_test_split(&x, &y).unwrap();

        assert_eq!(a_train.equals(&b_train), true);
    }

    #[test]
    fn test_stratified_split() {
        let (x, y) = sample_df();

        let strat = y.as_materialized_series().clone();

        let split = Split::new()
            .test_size(0.25)
            .random_state(42)
            .shuffle(true)
            .stratify(strat);

        let (_, x_test, _, y_test) =
            split.train_test_split(&x, &y).unwrap();

        assert_eq!(x_test.height(), 2);

        let positives = y_test
            .i32()
            .unwrap()
            .into_no_null_iter()
            .filter(|v| *v == 1)
            .count();

        assert!(positives <= 1);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let df = df![
            "f1" => &[1,2,3,4]
        ]
            .unwrap();

        let y = Series::new("target".into(), &[0,1]).into();

        let split = Split::new()
            .test_size(0.25);

        let result = split.train_test_split(&df, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_stratify_without_shuffle_error() {
        let (x, y) = sample_df();

        let strat = y.as_materialized_series().clone();

        let split = Split::new()
            .test_size(0.25)
            .shuffle(false)
            .stratify(strat);

        let result = split.train_test_split(&x, &y);

        assert!(result.is_err());
    }
}