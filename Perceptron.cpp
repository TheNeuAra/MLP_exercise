#include <iostream>
#include <vector>

class Perceptron {
    private:
    int input_dim;
    double learning_rate;
    int n_iters;
    std::vector<double> weights;

public:
    Perceptron(int input_dim, double learning_rate = 0.01, int n_iters = 1000) :
        input_dim(input_dim), learning_rate(learning_rate), n_iters(n_iters) {
        weights.resize(input_dim + 1, 0.0); // +1 for the bias term 用 pushback 可以吗？？
    }

    void train(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
        for (int it = 0; it < n_iters; ++it) {
            for (size_t i = 0; i < X.size(); ++i) {
                int prediction = predict(X[i]);
                int error = prediction - y[i];
                // Update weights and bias
                for (size_t j = 0; j < input_dim; ++j) {
                    weights[j] += learning_rate * error * X[i][j];
                }
                weights[input_dim] += learning_rate * error; // Update bias
            }
        }
    }

    int predict(const std::vector<double>& x) const {
        double sum = 0.0;
        for (size_t i = 0; i < input_dim; ++i) {
            sum += weights[i] * x[i];
        }
        sum += weights[input_dim]; // Add the bias term
        return (sum >= 0) ? 1 : 0;
    }


};

int main() {
    // 创建一个简单的二维数据集
    std::vector<std::vector<double>> X = { {2, 3}, {4, 6}, {4, 3}, {6, 8}, {1, 0.5}, {1, 1}, {3, 2}, {2, 1} };
    std::vector<int> y = { 0, 0, 0, 0, 1, 1, 1, 1 };

    // 创建感知器模型
    Perceptron perceptron(2);

    // 训练模型
    perceptron.train(X, y);

    // 测试模型
    std::vector<std::vector<double>> test_data = { {2, 3}, {1, 1}, {4, 5}, {6, 7} };
    for (const auto& sample : test_data) {
        int prediction = perceptron.predict(sample);
        std::cout << "Prediction for (" << sample[0] << ", " << sample[1] << "): " << prediction << std::endl;
    }

    return 0;
}
