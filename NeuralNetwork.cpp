#include "NeuralNetwork.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <numeric>
// 首先给所有的神经元的权重分配一个【0，1】的随机值，接受vector of input 作为 神经元层数的索引 eg, layers = {2，3，2}

/*NeuralNetwork::NeuralNetwork(const std::vector<int>& layers) : _layers(layers){
    std::srand(time(0));
    for(size_t i =1; i < _layers.size();  ++i){
        std::vector<std::vector<double>> AllLayerweight;
        //std::vector<std::vector<double>> Layerweight(_layers[i], std::vector<double>(_layers[i-1]+1)); 
        for(size_t j =1; j<_layers[i]; ++j){
            std::vector<double> Layerweight(_layers[i-1]+1);
            for(auto& neuroweight : Layerweight){
                
                neuroweight = randomWeight();
                
            }
            AllLayerweight.push_back(Layerweight);
        }
        _weights.push_back(AllLayerweight);
    }
   
}*/

NeuralNetwork::NeuralNetwork(const std::vector<int>& layers) : _layers(layers) {
    std::srand(std::time(0)); // Initialize random seed

    for (size_t i = 1; i < _layers.size(); ++i) { // 神经元的权重归属是1 index 不是 0 index
        std::vector<std::vector<double>> layerWeights(_layers[i], std::vector<double>(_layers[i - 1] + 1));//生成一个_layers[i]行_layers[i - 1] + 1列的矩阵，比如最开始的2行10列
        for (auto& neuronWeights : layerWeights) {
            for (auto& weight : neuronWeights) {
                weight = randomWeight();
            }
        }
        _weights.push_back(layerWeights);
    }
}
// 复现成功

//生成一个权重分布在0.0 到 1.0 之间的随机数的函数 RAND_MAX 是 《cstdlib》自带的宏定义32767
double NeuralNetwork::randomWeight() {
    return static_cast<double>((rand() / RAND_MAX) * 2.0 - 1.0);
}

// 同是也是为了sigmoid 激活函数的值域区间【0，1】考虑，所以才让随机权重分布在0，1之间
double NeuralNetwork::sigmoid(double x) { 
    return 1.0 / (1.0 + std::exp(-x));
}
/*
double findMax(const std::vector<double>& input) {
    double maxVal = input[0];
    for (size_t i = 1; i < input.size(); ++i) {
        if (input[i] > maxVal) {
            maxVal = input[i];
        }
    }
    return maxVal;
}

double NeuralNetwork::softMax(const std::vector<double>& input) {
    std::vector<double> output(input.size());
    double maxVal = findMax(input);  // 查找向量中的最大值
    double sum = 0.0;
    
    // 计算指数和它们的总和
    for(double val : input) {
        sum += std::exp(val - maxVal);
    }
    
    // 通过总和归一化
    for(size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - maxVal) / sum;
    }
    
    return output;
}

*/

//这一步是为了后面的反向传播算法的权重更新而定义的激活函数的导数
double NeuralNetwork::sigmoidDerivative(double x) {   
    return x * (1.0 - x);
}
 
//今天一定学会复现这个成员函数
std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) { // 比如输入的数据向量的第一个元素（向量）作为input vector
    _layerOutputs.clear();
    _layerOutputs.push_back(input);

    for (const auto& layerWeights : _weights) {
        std::vector<double> inputWithBias = _layerOutputs.back();
        inputWithBias.push_back(1.0); // Add bias term
        std::vector<double> output;

        for (const auto& neuronWeights : layerWeights) {
            double activation = 0.0;
            for (size_t i = 0; i < neuronWeights.size(); ++i) {
                activation += neuronWeights[i] * inputWithBias[i];
            } 
            output.push_back(sigmoid(activation));
        }
        _layerOutputs.push_back(output);
    }

    return _layerOutputs.back();
}
// ok 了 我自己再尝试复写一下

std::vector<double> NeuralNetwork::Forward(const std::vector<double>& Input){
    //首先Input是没有bias的，所以必须加上bias作为内部变量,后续再进行代码优化
    std::vector<double> Inputwithbias(Input.size());//必须得比input维度大1,而且最后一位必须是1 因为Bias藏在W向量里
    Inputwithbias.push_back(1.0);
    //接下来就是让每个神经元的权重都能乘以输入的值，这样就能得到激活值
    for(auto& layerweight : _weights){
        for(auto& neuralweight : layerweight){
            std::vector<double> ActVal = {0.0};   // 创建一列神经元的激活值
            for(size_t i =0; i< neuralweight.size(); ++i){ //对于每一个列神经元来说，他都有一串ActVal向量
                
                ActVal[i]= NeuralNetwork::sigmoid(Inputwithbias[i] * neuralweight[i]); // actval向量 储存列神经元的激活值

            }
            _layerOutputs.push_back(ActVal); //而这里的矩阵是储存神经元的所有ActVal向量
        }
    }
    return _layerOutputs.back();// 因为这个元素是矩阵而我们只在一其最后一行向量，所以。back（）就可以了
}
// 复现成功 完美复现，不仅如此，我还改进了代码并且减少了内存使用。 距离我神功大成还剩45%

void NeuralNetwork::backward(const std::vector<double>& input, const std::vector<double>& target, double learningRate) {
    _layerDeltas.clear(); // 之所以要每一次调用都清零，是因为这个_layerDeltas 的内容必须每一次Epoch都更新，直到最后一次调用
    _layerDeltas.resize(_layers.size());// 首先要把_layerDeltas 这个矩阵的大小变成固定的，虽然嵌入在内侧函数中可能会浪费资源，但是保障了后续索引的稳定性

    //首先计算MLP最后一层输出层的delta
    const auto& output = _layerOutputs.back(); // 首先找到最后一层的激活值向量 定义为 output
    std::vector<double> outputDelta(output.size()); // 然后再定义一个最后一层神经元的delta集合称为outputDelta, 每一个神经元有且只有一个对应的delta

    for (size_t i = 0; i < output.size(); ++i) {
        outputDelta[i] = (target[i] - output[i]) * NeuralNetwork::sigmoidDerivative(output[i]);// 计算最后一层的delta的公式而已
    }

    _layerDeltas.back() = outputDelta; // 先把这个outputDelta 安排到_layerDeltas对应的输出层的位置，后续再添加其他层的delta，最后再让_Weight[ijk]去根据3个for索引更新
 
// 首先计算第一层的layerdelta，尝试复现第一步简单的部分
/*
    const std::vector<double> Output = _layerOutputs.back(); 
    std::vector<double> Outputlayerdelta = _layerDeltas.back();

    for(size_t i = 0; i < Output.size(); ++i) {
        Outputlayerdelta[i] = (target[i]-Output[i])*sigmoidDerivative(Output[i]);
    }
    
    _layerDeltas.back() = Outputlayerdelta;
    //复现成功！！！！！！！！！！！！！！！！！！！！！！
*/

    //计算各个隐藏层的delta 并把他们计算出来之后依次安排进_layerDeltas 矩阵里面  公式 δj=σ ′(Zj)⋅ ∑k (δk ⋅ Wkj)
    for (int i = _layers.size() - 2; i > 0; --i) { // 索引 i 从后往前 i = _layers.size() - 2  对于从倒数第二层开始往前索引
        const auto& thisLayerWeights = _weights[i]; //  本层的对于下一层的神经元的权重矩阵
        const auto& nextLayerDelta = _layerDeltas[i + 1]; // 下一层所有神经元的delta向量
        std::vector<double> thisLayerdelta(_layerOutputs[i].size());

        for (size_t j = 0; j < _layerOutputs[i].size(); ++j) { // 对于每一个神经元来说
            double error = 0.0;
            for (size_t k = 0; k < thisLayerWeights.size(); ++k) { // 对于每个神经元所有的权重来讲
                error += nextLayerDelta[k] * thisLayerWeights[k][j]; //  ∑k (δk ⋅ Wkj) 对于下一层的所有神经元的delta与其对应的权重的乘积的和
            }
            thisLayerdelta[j] = error * NeuralNetwork::sigmoidDerivative(_layerOutputs[i][j]); // 对于每个神经元有且只有一个对应的delta 所以 layer delta是个矩阵 这一世为什么这句话在这个循环分支里
        }
        _layerDeltas[i] = thisLayerdelta; // 更新所有的神经元的delta （误差）
    }

    // 尝试复现反向传播的算法核心之 求反向传播后的 _layerdeltas . 公式 δj=σ ′(Zj)⋅ ∑k (δk ⋅ Wkj)
/*
    for(size_t i = _layers.size() - 2; i > 0; --i){              // i 遍历从倒数第二层神经元到正数第二个神经元
        
        std::vector<double> thislayerdelta(_layerDeltas[i].size()); // 此处经过修改，因为thislayerdelta 必须初始化成本层神经元的数量大小
        std::vector<std::vector<double>> nextlayerweight = _weights[i+1]; // nextlayerweight 必须是下一层神经元的权重矩阵

        for(size_t j = 0; j < _layerDeltas[i].size(); ++j){    // j 遍历 i 层的所有神经元 
            double Error = 0; //  ERROR 必须放在这里，因为每一个 i 层神经元都只有一个error放在外层的话会累计误差
            for(size_t k = 0 ; k < nextlayerweight[i].size(); ++k){ // k 遍历 i+1 层神经元
                Error += _layerDeltas[i+1][k] * nextlayerweight[k][j] ; // i+1层所有各个 delta 都分别取出来与i+1的各个神经元的第j个权重相乘
            }
            thislayerdelta[j] = Error * sigmoidDerivative(_layerOutputs[i][j]);
        }
        _layerDeltas[i] = thislayerdelta;
    }
    // 经过gpt检验，一次性复现了约 82 % 涵盖<逻辑 代码结构 细节>
    /*代码结构
相似度: 90%
对比: 两段代码都采用了从倒数第二层向前遍历隐藏层的方式，并在每一层计算delta值。这部分结构基本一致。
逻辑
相似度: 85%
对比: 两段代码都通过计算每个神经元的误差并用sigmoid导数计算delta值。然而，第二段代码在初始化Error的位置和遍历权重矩阵的索引上存在一些错误，这些逻辑错误会影响计算结果。
细节
相似度: 70%
对比: 原始代码在每一层神经元计算时，细节处理得更为准确，如Error的初始化和遍历权重矩阵的索引。复现代码在这些细节上存在一定的偏差。
    */
    // 复现成功 100%

 

    // 更新权重
    for (size_t i = 0; i < _weights.size(); ++i) {
        const auto& inputWithBias = (i == 0 ? input : _layerOutputs[i]); // 一定要注意input的分类, 只有在第一层的时候采用input，其他层都要用layerouput，因为这是在反向传播的语境下
        // 更新 ——weight 不需要重新输入input 而是用现成的参数完成反向传播 而不是一味的前反馈去求layerouput
        for (size_t j = 0; j < _weights[i].size(); ++j) {
            for (size_t k = 0; k < _weights[i][j].size(); ++k) {
                _weights[i][j][k] += learningRate * _layerDeltas[i + 1][j] * (k == _weights[i][j].size() - 1 ? 1.0 : inputWithBias[k]); // 一定要注意输入的 input 在最后一个元素一定是1 因为 bias储存在weights里面了
            }
        }
    }

    // 尝试复现 更新权重 的代码
    /*
        for(size_t i = 0; i < _weights.size(); ++i){ // i 遍历 所有列神经元
        const auto& inputwihtbias = (i == 0 ? input : _layerOutputs[i]);
        for(size_t j = 0; j < _weights[i].size();++j){ // j 遍历 i 列神经元的所有 nn

            for(size_t k = 0; k < _weights[i][j].size(); ++k){ // k 遍历 i 列 j 行  的所有权重
                _weights[i][j][k] += learningRate * _layerDeltas[i][j] * (k == _weights[i][j].size()-1 ? 1.0 : inputwihtbias[k]);
            }
        }
    }
    */

    // 浮现成功 100% 
}

 // 太简单了  用不着 复现
void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, double learningRate, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            forward(inputs[i]); // 这俩货的顺序不能反， 必须先用一个input【i】来激活所有的
            backward(inputs[i], targets[i], learningRate);
        } 
        std::cout << "Epoch " << epoch + 1 << " completed" << std::endl;
    }
}


// 保存模型到文件
void NeuralNetwork::saveModel(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (file.is_open()) {
            for (const auto& layer : _weights) {
                for (const auto& neurons : layer) {
                    for (double weight : neurons) {
                        file.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
                    }
                }
            }
            file.close();
        } else {
            std::cerr << "Unable to open file for writing: " << filename << std::endl;
        }
}

    // 从文件加载模型
void NeuralNetwork::loadModel(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (file.is_open()) {
            for (auto& layer : _weights) {
                for (auto& neuron : layer) {
                    for (double& weight : neuron) {
                        file.read(reinterpret_cast<char*>(&weight), sizeof(weight)); //因为 file.write 需要 const char* 类型的指针，而而不是是 double* 类型的指针。所以不能用const_cast<double>
                    }
                }
            }
            file.close();
        } else {
            std::cerr << "Unable to open file for reading: " << filename << std::endl;
        }
}