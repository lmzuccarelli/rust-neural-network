use ai::activation::SIGMOID;
use ai::network::Network;

mod ai;

fn main() {
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
    let mut network = Network::new(vec![2, 3, 1], 0.5, SIGMOID);

    network.load("res.json".to_string());
    network.train(inputs, targets, 10000);

    println!(
        "0 and 0 : {:?} ",
        network.forward_propagation(vec![0.0, 0.0])
    );
    println!(
        "0 and 1 : {:?} ",
        network.forward_propagation(vec![0.0, 1.0])
    );
    println!(
        "1 and 0 : {:?} ",
        network.forward_propagation(vec![1.0, 0.0])
    );
    println!(
        "1 and 1 : {:?} ",
        network.forward_propagation(vec![1.0, 1.0])
    );

    network.save("res.json".to_string());
}
