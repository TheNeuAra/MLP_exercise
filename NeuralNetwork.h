/*
这段代码实现了一个具有多层隐藏层的神经网络。
可以通过修改layers变量来配置隐藏层的数量和每层的神经元数量。
这个示例使用XOR问题的数据来训练和测试神经网络。*/

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <string>

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layers);
    std::vector<double> forward(const std::vector<double>& input);
    std::vector<double> Forward(const std::vector<double>& input);
    void backward(const std::vector<double>& input, const std::vector<double>& target, double learningRate);
    void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, double learningRate, int epochs);
    void saveModel(const std::string& filename);
    void loadModel(const std::string& filename);

private:
    double sigmoid(double x);
    double sigmoidDerivative(double x);
    double randomWeight();
    //double softMax(std::vector<double> Input);

    std::vector<int> _layers;
    std::vector<std::vector<std::vector<double>>> _weights;
    std::vector<std::vector<double>> _layerOutputs;
    std::vector<std::vector<double>> _layerDeltas;
};

#endif //NEURONETWORK_H
