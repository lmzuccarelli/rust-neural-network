use std::f64::consts::E;

#[derive(Clone)]
pub struct Activation<'a> {
    pub function: &'a dyn Fn(f64) -> f64,
    pub derivative: &'a dyn Fn(f64) -> f64,
}

#[allow(dead_code)]
pub const IDENTITY: Activation = Activation {
    function: &|x| x,
    derivative: &|_| 1.0,
};

pub const SIGMOID: Activation = Activation {
    function: &|x| 1.0 / (1.0 + E.powf(-x)),
    derivative: &|x| x * (1.0 - x),
};

#[allow(dead_code)]
pub const TANH: Activation = Activation {
    function: &|x| x.tanh(),
    derivative: &|x| 1.0 - (x.powi(2)),
};

#[allow(dead_code)]
pub const RELU: Activation = Activation {
    function: &|x| x.max(0.0),
    derivative: &|x| if x > 0.0 { 1.0 } else { 0.0 },
};

#[cfg(test)]
mod tests {
    // this brings everything from parent's scope into this scope
    use super::*;
    use crate::ai::network::*;

    #[test]
    fn activation_sigmoid_pass() {
        let inputs = vec![1.0, 1.0];
        let targets = vec![1.0];
        let mut net = Network::new(vec![2, 2, 1], 0.5, SIGMOID);
        let res = net.forward_propagation(inputs);
        net.back_propagation(res, targets);
    }

    #[test]
    fn activation_identity_pass() {
        let inputs = vec![1.0, 1.0];
        let targets = vec![1.0];
        let mut net = Network::new(vec![2, 2, 1], 0.5, IDENTITY);
        let res = net.forward_propagation(inputs);
        net.back_propagation(res, targets);
    }

    #[test]
    fn activation_tanh_pass() {
        let inputs = vec![1.0, 1.0];
        let targets = vec![1.0];
        let mut net = Network::new(vec![2, 2, 1], 0.5, TANH);
        let res = net.forward_propagation(inputs);
        net.back_propagation(res, targets);
    }

    #[test]
    fn activation_relu_pass() {
        let inputs = vec![1.0, 1.0];
        let targets = vec![1.0];
        let mut net = Network::new(vec![2, 2, 1], 0.5, RELU);
        let res = net.forward_propagation(inputs);
        net.back_propagation(res, targets);
    }
}
