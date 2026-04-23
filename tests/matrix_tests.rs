#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use custom_math_ml_stuff::matrix::Matrix;
    use custom_math_ml_stuff::errors::CommonError::CommonError::DimensionMismatch;
    use custom_math_ml_stuff::errors::matrix_error::MatrixError;

    fn setup_matrix(rows: usize, cols: usize, value: f64) -> Matrix<f64> {
        let mut m = Matrix::new(rows, cols);
        m.fill_matrix(value);
        m
    }

    #[test]
    fn test_new_from_vec_success() {
        let m = Matrix::from_vec(2, 3, vec![1f64, 2f64, 3f64, 4f64, 5f64, 6f64]);
        assert_eq!(m.unwrap().matrix, vec![1f64, 2f64, 3f64, 4f64, 5f64, 6f64])
    }

    #[test]
    fn test_new_from_vec_fails() {
        let m = Matrix::from_vec(2, 3, vec![1f64, 2f64, 3f64, 4f64, 5f64]);
        assert!(matches!(m, Err(MatrixError::CommonError(DimensionMismatch))))
    }
    #[test]
    fn test_new_matrix() {
        let m: Matrix<f64> = Matrix::new(2, 3);

        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.matrix.len(), 6);
        assert_eq!(m.sum_matrix(), 0.0);
    }

    #[test]
    fn test_fill_matrix() {
        let mut m = Matrix::new(2, 2);
        m.fill_matrix(5.0);

        assert_eq!(m.matrix, vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_clear_matrix() {
        let mut m = setup_matrix(2, 2, 4.0);
        m.clear_matrix();

        assert_eq!(m.matrix, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_scale_matrix() {
        let mut m = setup_matrix(2, 2, 2.0);
        m.scale_matrix(3.0);

        assert_eq!(m.matrix, vec![6.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn test_add_matrix_success() {
        let mut a = setup_matrix(2, 2, 1.0);
        let b = setup_matrix(2, 2, 2.0);

        let res = a.add_matrix(&b);

        assert!(res.is_ok());
        assert_eq!(a.matrix, vec![3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_add_matrix_dimension_error() {
        let mut a:Matrix<f64> = Matrix::new(2, 2);
        let b = Matrix::new(3, 3);

        let res = a.add_matrix(&b);

        assert!(matches!(res, Err(MatrixError::CommonError(DimensionMismatch))));
    }

    #[test]
    fn test_sub_matrix_success() {
        let mut a = setup_matrix(2, 2, 5.0);
        let b = setup_matrix(2, 2, 2.0);

        let res = a.sub_matrix(&b);

        assert!(res.is_ok());
        assert_eq!(a.matrix, vec![3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_sub_matrix_dimension_error() {
        let mut a: Matrix<f64> = Matrix::new(2, 2);
        let b = Matrix::new(3, 3);

        let res = a.sub_matrix(&b);

        assert!(matches!(res, Err(MatrixError::CommonError(DimensionMismatch))));
    }

    #[test]
    fn test_sum_matrix() {
        let m = setup_matrix(2, 3, 2.0);

        assert_eq!(m.sum_matrix(), 12.0);
    }

    #[test]
    fn test_mul_matrix_success() {
        let mut a = Matrix::new(2, 2);
        a.matrix = vec![1.0, 2.0, 3.0, 4.0];

        let mut b = Matrix::new(2, 2);
        b.matrix = vec![5.0, 6.0, 7.0, 8.0];

        let res = a.mul_matrix(&b).unwrap();

        assert_eq!(res.rows, 2);
        assert_eq!(res.cols, 2);
        assert_eq!(res.matrix, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_mul_matrix_dimension_error() {
        let a: Matrix<f64> = Matrix::new(2, 3);
        let b = Matrix::new(4, 2);

        let res = a.mul_matrix(&b);

        assert!(matches!(res, Err(MatrixError::CommonError(DimensionMismatch))));
    }

    #[test]
    fn test_mul_diff_dimensions_success() {
        let a = Matrix::from_vec(2, 3, vec![1f64, 2f64, 3f64, 4f64, 5f64, 6f64]);
        let b = Matrix::from_vec(3, 2, vec![7f64, 8f64, 9f64, 10f64, 11f64, 12f64]);

        assert_eq!(
            a.unwrap().mul_matrix(&b.unwrap()).unwrap().matrix,
            vec![58f64, 64f64, 139f64, 154f64]
        )
    }

    #[test]
    fn test_mul_1() {
        let a = Matrix::from_vec(
            3,
            3,
            vec![2f64, -1f64, 3f64, 0f64, 4f64, 1f64, -2f64, 5f64, 2f64],
        );
        let b = Matrix::from_vec(
            3,
            3,
            vec![1f64, 2f64, 0f64, -1f64, 3f64, 4f64, 2f64, -1f64, 1f64],
        );

        assert_eq!(
            a.unwrap().mul_matrix(&b.unwrap()).unwrap().matrix,
            vec![9f64, -2f64, -1f64, -2f64, 11f64, 17f64, -3f64, 9f64, 22f64]
        )
    }

    #[test]
    fn test_mul_2() {
        let a = Matrix::from_vec(
            3,
            4,
            vec![
                1f64, 2f64, -1f64, 3f64, 0f64, -2f64, 4f64, 1f64, 3f64, 1f64, 0f64, -1f64,
            ],
        );
        let b = Matrix::from_vec(
            4,
            3,
            vec![
                2f64, -1f64, 0f64, 1f64, 3f64, 2f64, -2f64, 4f64, 1f64, 0f64, 5f64, -3f64,
            ],
        );

        assert_eq!(
            a.unwrap().mul_matrix(&b.unwrap()).unwrap().matrix,
            vec![6f64, 16f64, -6f64, -10f64, 15f64, -3f64, 7f64, -5f64, 5f64]
        )
    }

    #[test]
    fn test_mul_10x3_3x4() {
        let a = Matrix::from_vec(
            10,
            3,
            vec![
                1f64, 2f64, 3f64, 4f64, 5f64, 6f64, 7f64, 8f64, 9f64, 10f64, 11f64, 12f64, 13f64,
                14f64, 15f64, 16f64, 17f64, 18f64, 19f64, 20f64, 21f64, 22f64, 23f64, 24f64, 25f64,
                26f64, 27f64, 28f64, 29f64, 30f64,
            ],
        );

        let b = Matrix::from_vec(
            3,
            4,
            vec![
                1f64, 2f64, 3f64, 4f64, 5f64, 6f64, 7f64, 8f64, 9f64, 10f64, 11f64, 12f64,
            ],
        );

        let expected = vec![
            38f64, 44f64, 50f64, 56f64, 83f64, 98f64, 113f64, 128f64, 128f64, 152f64, 176f64,
            200f64, 173f64, 206f64, 239f64, 272f64, 218f64, 260f64, 302f64, 344f64, 263f64, 314f64,
            365f64, 416f64, 308f64, 368f64, 428f64, 488f64, 353f64, 422f64, 491f64, 560f64, 398f64,
            476f64, 554f64, 632f64, 443f64, 530f64, 617f64, 704f64,
        ];

        assert_eq!(a.unwrap().mul_matrix(&b.unwrap()).unwrap().matrix, expected);
    }

    #[test]
    fn test_det_matrix_success() {
        let n = Matrix::from_vec(2, 2, vec![4.0, 6.0, 3.0, 8.0]);
        assert_eq!(n.unwrap().determinant().unwrap(), 14.0)
    }

    #[test]
    fn test_det_matrix_success_2() {
        let n = Matrix::from_vec(3, 3, vec![6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0]);
        assert_eq!(n.unwrap().determinant().unwrap(), -306.0)
    }

    #[test]
    fn test_det_matrix_fail() {
        let n = Matrix::from_vec(2, 3, vec![6.0, 1.0, 1.0, 4.0, -2.0, 5.0]);
        assert!(matches!(
            n.unwrap().determinant(),
            Err(MatrixError::NotSquaredMatrix)
        ))
    }

    #[test]
    fn test_inverse_matrix_fail() {
        let n = Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert!(matches!(
            n.unwrap().inverse_matrix(),
            Err(MatrixError::NotInverseable)
        ))
    }

    #[test]
    fn test_inverse_matrix() {
        let n = Matrix::from_vec(2, 2, vec![1.0, 2.0, 5.00, 6.0]);

        assert_eq!(
            n.unwrap().inverse_matrix().unwrap().matrix,
            vec![-1.5, 0.5, 1.25, -0.25]
        )
    }

    #[test]
    fn test_inverse_matrix_2() {
        let n = Matrix::from_vec(3, 3, vec![1.0, 2.0, 6.0, 4.0, 4.0, 8.0, 3.0, 3.0, -1.0]).unwrap();

        let expected = vec![
            -1.0,
            0.7142857142857143,
            -0.2857142857142857,
            1.0,
            -0.6785714286,
            0.5714285714,
            0.0,
            0.1071428571,
            -0.1428571429
        ];

        for (a, b) in n.inverse_matrix().unwrap().matrix.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_transpose_matrix_shape() {
        let mut m = Matrix::new(2, 3);
        m.matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let a = m.transposed_matrix();

        assert_eq!(a.rows, 3);
        assert_eq!(a.cols, 2);
    }
}
