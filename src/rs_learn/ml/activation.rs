use num::{Float, Num, Signed};

pub struct Activation<T: Copy + Num + Signed>{
    pub function: Box<dyn Fn(T) -> T>,
    pub derivative: Box<dyn Fn(T) -> T>,
}

pub fn sigmoid<T: Float + Signed>() -> Activation<T>{
    Activation{
        function: Box::new(|x| T::one() / (T::one() + (-x).exp())),
        derivative: Box::new(|x| x * (T::one() - x))
    }
}

pub fn tan_h<T: Float + Signed>() -> Activation<T> {
    Activation{
        function:  Box::new(|x| {
            let two = T::one() + T::one();
            two / (T::one() + (-two * x).exp()) - T::one()
        }),
        derivative:  Box::new(|x| {
            let two = T::one() + T::one();
            (two * two) * ((-two*x).exp() / (two * (-two*x).exp() + ((-two*x).exp()).powi(2) + T::one() ))
        })
    }
}

pub fn relu<T: Num + Signed + PartialOrd + Copy>() -> Activation<T> {
    Activation{
        function:  Box::new(|x| if x > T::zero() { x } else { T::zero() }),
        derivative:  Box::new(|x| if x > T::zero() { T::one() } else { T::zero() })
    }
}

pub fn leaky_relu<T: Num + Signed + PartialOrd + Copy + 'static>(alpha: T) -> Activation<T> {
    Activation {
        function: Box::new(move |x| {
            if x > T::zero() {
                x
            } else {
                alpha * x
            }
        }),
        derivative: Box::new(move |x| {
            if x > T::zero() {
                T::one()
            } else {
                alpha
            }
        }),
    }
}