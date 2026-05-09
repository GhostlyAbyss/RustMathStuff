#[cfg(test)]
mod tests {
    use custom_math_ml_stuff::rs_learn::linear_regression::*;
    use polars::prelude::*;
    use custom_math_ml_stuff::matrix::Matrix;
    use custom_math_ml_stuff::rs_learn::split::Split;

    #[test]
    fn test_linear_regression_works() -> PolarsResult<()> {
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some("tests/test_data/housing.csv".into()))?
            .finish()?;

        let x = &df.select(["median_income"])?;
        let y = &df["median_house_value"];

        let (x_train_df, x_test_df, y_train_col, y_test_col) = Split::new()
            .test_size(0.2)
            .random_state(42)
            .train_test_split(x, y)
            .unwrap();

        let x_train = df_to_matrix(&x_train_df);
        let y_train = col_to_vec(&y_train_col);
        let x_test = df_to_matrix(&x_test_df);
        let y_test = col_to_vec(&y_test_col);

        let mut model = LinearRegression::new().n_jobs(-1);

        model.fit(&x_train, &y_train).unwrap();

        let r2 = model.score(&x_test, &y_test).unwrap();

        assert!(r2.is_finite());
        assert!(r2 <= 1.0);
        assert!(r2 >= -1.0);

        Ok(())
    }

    fn col_to_vec(col: &Column) -> Vec<f64> {
        col.f64()
            .unwrap()
            .into_no_null_iter()
            .collect()
    }

    fn df_to_matrix(df: &DataFrame) -> Matrix<f64> {
        let rows = df.height();
        let cols = df.width();

        let mut data = Vec::with_capacity(rows * cols);

        for col in df.columns() {
            let ca = col.f64().unwrap();
            data.extend(ca.into_no_null_iter());
        }

        Matrix::from_vec(rows, cols, data).unwrap()
    }

}
