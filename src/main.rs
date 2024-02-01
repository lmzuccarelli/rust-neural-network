use ai::activation::SIGMOID;
use ai::network::Network;
use std::path::Path;

mod ai;

fn main() {
    /*

      7 Segment display

          a
        #####
      f #   # b
        #   #
        ##g##
        #   #
      e #   # c
        #####
          d

      inputs      outputs      display
      ------      -----------  -------
      0 0 0       a,b,c,d,e,f  0
      0 0 1       b,c          1
      0 1 0       a,b,d,e,g    2
      0 1 1       a,b,c,d,g    3
      1 0 0       b,c,f,g      4
      1 0 1       a,c,d,f,g    5
      1 1 0       a,c,d,e,f,g  6
      1 1 1       a,b,c        7

    */

    let inputs = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 0.0],
        vec![1.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
        vec![1.0, 1.0, 1.0],
    ];

    // Led segment outputs
    //       a    b    c    d    e    f    g
    let targets = vec![
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ];

    let mut network = Network::new(vec![3, 4, 7], 0.2, SIGMOID);
    let file_path = "model.json";
    let path = Path::new(file_path);
    if path.exists() {
        network.load("model.json".to_string());
    }

    network.train(inputs, targets, 10000);

    println!(
        "0 0 0 : {:?} ",
        network.forward_propagation(vec![0.0, 0.0, 0.0])
    );
    println!(
        "0 0 1 : {:?} ",
        network.forward_propagation(vec![0.0, 0.0, 1.0])
    );
    println!(
        "0 1 0 : {:?} ",
        network.forward_propagation(vec![0.0, 1.0, 0.0])
    );
    println!(
        "0 1 1 : {:?} ",
        network.forward_propagation(vec![0.0, 1.0, 1.0])
    );

    println!(
        "1 0 0 : {:?} ",
        network.forward_propagation(vec![1.0, 0.0, 0.0])
    );
    println!(
        "1 0 1 : {:?} ",
        network.forward_propagation(vec![1.0, 0.0, 1.0])
    );
    println!(
        "1 1 0 : {:?} ",
        network.forward_propagation(vec![1.0, 1.0, 0.0])
    );
    println!(
        "1 1 1 : {:?} ",
        network.forward_propagation(vec![1.0, 1.0, 1.0])
    );

    network.save("model.json".to_string());
}
