#include <iostream>
#include <vector>
#include <armadillo>
#include <cmath>
#include <filesystem>
#include <cassert>
#include <fstream>
/**
 * \file
 * \brief Header file with Perceptron model
 */


/**
 * @class Perceptron
 * @brief A simple multi-layer perceptron implementation using Armadillo for matrix operations.
 *
 * Provides methods for training, prediction, saving/loading the model, and image-based data handling.
 */

class Perceptron {
    private:
        int input_size;                          ///< Number of input features
        std::vector<int> hidden_layers_size;    ///< Sizes of each hidden layer
        int output_size;                         ///< Number of output classes
        double learning_rate;                    ///< Learning rate for training
        int epochs;                              ///< Number of epochs for training

        std::vector<arma::mat> weights_input_hidden;  ///< Weight matrices between input and hidden layers
        arma::mat weights_output_hidden;             ///< Weight matrix for the hidden-to-output layer
        std::vector<arma::mat> bias_input_hidden;    ///< Bias vectors for each hidden layer

        std::vector<arma::mat> hidden_layer_input;   ///< Linear combination results for each hidden layer
        std::vector<arma::mat> hidden_layer_output;  ///< Activation outputs for each hidden layer
        arma::mat final_hidden_layer_output;         ///< Activation output of the last hidden layer

        arma::mat X;                                 ///< Last input passed to forward

        /**
         * @brief Creates a one-hot encoded vector for a label.
         * @param y Class label index
         * @param n Total number of classes (default = 10)
         * @return One-hot encoded row vector of length n
         */
        arma::mat one_hot(int y, int n = 10);

        /**
         * @brief Checks if the given image file path is valid.
         * @param path File path to an image
         * @return True if file exists and is readable, false otherwise
         */
        bool image_path_correct(const std::string& path);
    public:
        /**
         * @brief Constructs a perceptron with specified architecture.
         * @param input_size Number of input neurons (features)
         * @param hidden_layers_size Vector specifying number of neurons in each hidden layer
         * @param output_size Number of output neurons (classes)
         * @param learning_rate Learning rate for weight updates
         * @param epochs Number of epochs to train
         */
        Perceptron(int input_size = 28 * 28, std::vector<int> hidden_layers_size = {400, 256, 128}, int output_size = 10, double learning_rate = 0.01, int epochs = 1);

        /**
         * @brief Constructs a perceptron and loads weights/biases from file.
         * @param path Path to the saved model file
         */
        Perceptron(std::string path);

        /**
         * @brief Initializes random weights for all layers.
         */
        void gen_weights();

        /**
         * @brief Initializes random biases for all hidden layers.
         */
        void bias_gen();

        /**
         * @brief Applies the tanh activation function element-wise.
         * @param x Input matrix/vector
         * @return Matrix/vector after tanh activation
         */
        arma::mat tanh_activation(const arma::mat& x);
        /**
         * @brief Computes the derivative of the tanh activation.
         * @param x Input matrix/vector (pre-activation)
         * @return Matrix/vector of derivatives
         */
        arma::mat tanh_derivative(const arma::mat& x);

        /**
         * @brief Applies the softmax function to the input.
         * @param x Input matrix/vector
         * @return Matrix/vector of probabilities summing to 1 across rows
         */
        arma::mat softmax(const arma::mat& x);

        /**
         * @brief Performs a forward pass through the network.
         * @param X Input features as a row vector or matrix
         * @return Output probabilities as a row vector or matrix
         */
        arma::mat forward(const arma::mat& X);    

        /**
         * @brief Performs backpropagation for a single example.
         * @param y_train True class label index
         */
        void backprop(const int y_train);

        /**
         * @brief Trains the model on a dataset.
         * @param X_train Vector of input feature matrices
         * @param y_train Vector of class labels
         * @param epochs Number of epochs to train
         * @param learning_rate Learning rate for training
         */
        void train(const std::vector<arma::mat>& X_train, const std::vector<int>& y_train, int epochs, double learning_rate);

        /**
         * @brief Predicts the class label for given input features.
         * @param X Input features as a row vector
         * @return Predicted class index
         */
        int predict(const arma::mat& X);

        /**
         * @brief Predicts the class of an image.
         * @param path File path to the image
         * @return Predicted class index, or -1 if file is invalid
         */
        int predict_image(std::string path);

        /**
         * @brief Saves the model structure, weights, and biases to file.
         * @param path File path to save the model
         */
        void save_model(const std::string& path);

        /**
         * @brief Loads model structure, weights, and biases from file.
         * @param path File path to load the model from
         */
        void load_model(const std::string& path);

        /**
         * @brief Reads a single image file into a matrix.
         * @param path File path to the image
         * @param image_bbp Bytes per pixel (default = 1)
         * @return Image data as an Armadillo matrix
         */
        arma::mat read_image(const std::string& path, int image_bbp = 1);

        /**
         * @brief Reads multiple images into a vector of matrices.
         * @param pathes Vector of image file paths
         * @return Vector of image data matrices
         */
        std::vector<arma::mat> read_images(const std::vector<std::string>& pathes);

        /**
         * @brief Trains on images specified by a file containing paths and labels.
         * @param path_to_file File path listing image paths and labels
         * @param cur_epochs Number of epochs (-1 to use default)
         * @param cur_learning_rate Learning rate (-1 to use default)
         */
        void train_on_specific_images(std::string path_to_file, int cur_epochs = -100, double cur_learning_rate = -100);

        /** @name Getters */
        ///@{
        const std::vector<int> get_hidden_layers_size() const { return hidden_layers_size; }
        const std::vector<arma::mat> get_weights_input_hidden() const { return weights_input_hidden; }
        const arma::mat& get_weights_output_hidden() const { return weights_output_hidden; }
        const std::vector<arma::mat> get_bias_input_hidden() const { return bias_input_hidden; }
        
        const std::vector<arma::mat> get_hidden_layer_input() const { return hidden_layer_input; }
        const std::vector<arma::mat> get_hidden_layer_output() const { return hidden_layer_output; }
        const arma::mat get_final_hidden_layer_output() const { return final_hidden_layer_output; }
        const int get_input_size() { return input_size; }
        const int get_output_size() { return output_size; }
        ///@}
        
        bool error_output = true;
};
