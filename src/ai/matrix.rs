use rand::{thread_rng, Rng};
use std::fmt::{Debug, Formatter, Result};

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn zero(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    pub fn random(rows: usize, cols: usize) -> Matrix {
        let mut rng = thread_rng();
        let mut res = Matrix::zero(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                res.data[i][j] = rng.gen::<f64>() * 2.0 - 1.0;
            }
        }
        res
    }

    pub fn multiply(&mut self, target: &Matrix) -> Matrix {
        if self.cols != target.rows {
            panic!("Can't multiply a matrix of incorrect dimension")
        }

        let mut res = Matrix::zero(self.rows, target.cols);
        for i in 0..self.rows {
            for j in 0..target.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * target.data[k][j]
                }
                res.data[i][j] = sum;
            }
        }
        res
    }

    pub fn add(&mut self, target: &Matrix) -> Matrix {
        if self.rows != target.rows || self.cols != target.cols {
            panic!("Can't add a matrix of incorrect dimension")
        }

        let mut res = Matrix::zero(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] + target.data[i][j]
            }
        }
        res
    }

    pub fn dot_multiply(&mut self, target: &Matrix) -> Matrix {
        if self.rows != target.rows || self.cols != target.cols {
            panic!("Can't dot_multiply a matrix of incorrect dimension")
        }

        let mut res = Matrix::zero(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] * target.data[i][j]
            }
        }
        res
    }

    pub fn subtract(&mut self, target: &Matrix) -> Matrix {
        if self.rows != target.rows || self.cols != target.cols {
            panic!("Can't subtract a matrix of incorrect dimension")
        }

        let mut res = Matrix::zero(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] - target.data[i][j]
            }
        }
        res
    }

    pub fn from(data: Vec<Vec<f64>>) -> Matrix {
        Matrix {
            rows: data.len(),
            cols: data[0].len(),
            data,
        }
    }

    pub fn map(&mut self, function: &dyn Fn(f64) -> f64) -> Matrix {
        Matrix::from(
            (self.data)
                .clone()
                .into_iter()
                .map(|row| row.into_iter().map(|value| function(value)).collect())
                .collect(),
        )
    }

    pub fn transpose(&mut self) -> Matrix {
        let mut res = Matrix::zero(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[j][i] = self.data[i][j];
            }
        }
        res
    }
}

impl Debug for Matrix {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "Matrix {{\n{}\n}}",
            (&self.data)
                .into_iter()
                .map(|row| "  ".to_string()
                    + &row
                        .into_iter()
                        .map(|value| value.to_string())
                        .collect::<Vec<String>>()
                        .join(" "))
                .collect::<Vec<String>>()
                .join("\n")
        )
    }
}

#[cfg(test)]
mod tests {
    // this brings everything from parent's scope into this scope
    use super::*;

    #[test]
    fn test_matrix_zero_pass() {
        let res = Matrix::zero(2, 2);
        assert!(res.rows == 2);
        assert!(res.cols == 2);
        assert!(res.data[0][0] == 0.0);
        assert!(res.data[1][1] == 0.0);
    }

    #[test]
    fn test_matrix_random_pass() {
        let res = Matrix::random(2, 2);
        assert!(res.rows == 2);
        assert!(res.cols == 2);
        assert!(res.data[0][0] != 0.0);
        assert!(res.data[1][1] != 0.0);
    }

    #[test]
    fn test_matrix_multiply_pass() {
        let mut a = Matrix::random(2, 2);
        let b = Matrix::random(2, 2);
        let res = a.multiply(&b);
        assert!(res.rows == 2);
        assert!(res.cols == 2);
        assert!(res.data[0][0] != 0.0);
        assert!(res.data[1][1] != 0.0);
    }

    #[test]
    fn test_matrix_add_pass() {
        let mut a = Matrix::random(2, 2);
        let b = Matrix::random(2, 2);
        let res = a.add(&b);
        assert!(res.rows == 2);
        assert!(res.cols == 2);
        assert!(res.data[0][0] != 0.0);
        assert!(res.data[1][1] != 0.0);
    }

    #[test]
    fn test_matrix_dot_multiply_pass() {
        let mut a = Matrix::random(2, 2);
        let b = Matrix::random(2, 2);
        let res = a.dot_multiply(&b);
        assert!(res.rows == 2);
        assert!(res.cols == 2);
        assert!(res.data[0][0] != 0.0);
        assert!(res.data[1][1] != 0.0);
    }

    #[test]
    fn test_matrix_subtract_pass() {
        let mut a = Matrix::random(2, 2);
        let b = Matrix::random(2, 2);
        let res = a.subtract(&b);
        assert!(res.rows == 2);
        assert!(res.cols == 2);
        assert!(res.data[0][0] != 0.0);
        assert!(res.data[1][1] != 0.0);
    }

    #[test]
    fn test_matrix_transpose_pass() {
        let mut a = Matrix::random(2, 2);
        let res = a.transpose();
        assert!(res.rows == 2);
        assert!(res.cols == 2);
    }
}
