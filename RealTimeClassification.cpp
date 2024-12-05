#include <opencv2/opencv.hpp>
#include "NeuralNetwork.h"
#include <vector>
#include <iostream>

// Function to preprocess the frame
std::vector<double> preprocessImage(cv::Mat& frame) {
    cv::Mat gray, resized;

    // Convert to grayscale
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Resize to 28x28
    cv::resize(gray, resized, cv::Size(28, 28));

    // Normalize pixel values to [0, 1]
    resized.convertTo(resized, CV_64F, 1.0 / 255);

    // Flatten the image to a 784-dimensional vector
    std::vector<double> input(resized.begin<double>(), resized.end<double>());

    return input;
}

int main() {
    // Define the neural network structure and create the network
    std::vector<int> layers = {784, 128, 64, 10};
    NeuralNetwork nn(layers);

    // XOR problem training data (for demonstration purposes, replace with your actual data)
    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    std::vector<std::vector<double>> targets = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    // Training parameters
    double learningRate = 0.1;
    int epochs = 10000;

    // Train the neural network
    nn.train(inputs, targets, learningRate, epochs);

    // Open the default camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    //load my Weights file
    nn.loadModel("weights");
    cv::Mat frame;
    while (true) {
        // Capture a new frame from the camera
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Could not grab a frame." << std::endl;
            break;
        }

        // Preprocess the frame
        std::vector<double> input = preprocessImage(frame);

        // Use the trained neural network to predict the label
        std::vector<double> output = nn.forward(input);

        // Find the predicted label
        int predictedLabel = std::distance(output.begin(), std::max_element(output.begin(), output.end()));

        // Display the predicted label on the frame
        cv::putText(frame, "Predicted: " + std::to_string(predictedLabel), cv::Point(30, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        // Show the frame
        cv::imshow("Camera", frame);

        // Break the loop if the user presses the 'q' key
        if (cv::waitKey(30) >= 0 && (char)cv::waitKey(30) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
