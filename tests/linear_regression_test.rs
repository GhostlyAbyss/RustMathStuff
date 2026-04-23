#[cfg(test)]
mod tests {
    use custom_math_ml_stuff::rs_learn::linear_regression::*;
    use polars::prelude::*;

    #[test]
    fn test() -> PolarsResult<()> {
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some("tests/test_data/housing.csv".into()))?
            .finish()?;

        let X = &df["median_income"];
        let y = &df["median_house_value"];
        println!("{:?}", X.head(Some(1)));
        println!("{:?}", y.head(Some(1)));
        Ok(())
    }
}
