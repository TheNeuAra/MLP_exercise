#include <iostream>
#include <vector>
#include "NeuralNetwork.h"

int main() {
    std::vector<int> layers = {2, 20, 20, 20, 20, 1}; // Input layer, two hidden layers, and output layer
    NeuralNetwork nn(layers);

    // XOR problem training data
    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0},
        {5.0, 1.0},
        {100.0, 1.0},
        {0.1, 0.019}
    };
    std::vector<std::vector<double>> targets = {
        {1.0},
        {0.0},
        {0.0},
        {0.0},
        {0.0},
        {0.0},
        {1.0}
    };

    //nn.loadModel("Weights.bin");

    double learningRate = 0.3;
    int epochs = 5000;

    nn.train(inputs, targets, learningRate, epochs);

    for (const auto& input : inputs) {
        std::vector<double> output = nn.forward(input);
        std::cout << "Input: ";
        for (double val : input) {
            std::cout << val << " ";
        }
        std::cout << "Output: ";
        for (double val : output) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    nn.saveModel("Weights.bin");

    return 0;
}
