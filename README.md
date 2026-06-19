[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/OcE5Fe4c)

# Exploring Hyperparameters

    I chose the following values:
    leaky_relu, elu, relu, selu and swish
    The results are:

    | activation function | best val_accuracy | best val_loss | n epochs  |
    |---------------------|-------------------|---------------|-----------|
    | elu                 | 0.9375            | 0.2390        | 19        |
    | relu                | 0.9297            | 0.2897        | 30        |
    | selu                | 0.8828            | 0.2675        | 13        |
    | swish               | 0.9062            | 0.4380        | 18        |
    | leaky_relu          | 0.9531            | 0.2126        | 20        |
    |_____________________|___________________|_______________|___________|

Elu
![Elu Graph](01-hyperparameters/plots/elu1.png "Elu")

Relu
![Relu Graph](01-hyperparameters/plots/relu1.png "Relu")

Selu
![Selu Graph](01-hyperparameters/plots/selu1.png "Selu")

Swish
![Swish Graph](01-hyperparameters/plots/swish1.png "Swish")

Leaky Relu
![Leaky Relu Graph](01-hyperparameters/plots/leaky_relu1.png "Leaky Relu")
