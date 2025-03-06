#include <iostream>
#include <vector>
#include <armadillo>
#include <cmath>
#include <filesystem>
#include <cassert>
#include <fstream>


class Perceptron {
    public:
        Perceptron(int input_size = 28 * 28, std::vector<int> hidden_layers_size = {400, 256, 128}, int output_size = 10, double learning_rate = 0.01, int epochs = 1);
        
        void gen_weights();
        void bias_gen();
        arma::mat tanh_activation(const arma::mat& x);
        arma::mat tanh_derivative(const arma::mat& x);
        arma::mat softmax(const arma::mat& x);
        arma::mat forward(const arma::mat& X);    
        void backprop(const int y_train);
        void train(const std::vector<arma::mat>& X_train, const std::vector<int>& y_train, int epochs, double learning_rate);
        int predict(const arma::mat& X);
        void save_model(const std::string& path);
        void load_model(const std::string& path);

        arma::mat read_image(const std::string& path, int image_bbp = 1);
        std::vector<arma::mat> read_images(const std::vector<std::string>& pathes);
        void train_on_specific_images(std::string path_to_file);
    private:
        int input_size;
        std::vector<int> hidden_layers_size;
        int output_size;
        double learning_rate;
        int epochs;
        
        std::vector<arma::mat> weights_input_hidden;
        arma::mat weights_output_hidden;
        std::vector<arma::mat> bias_input_hidden;
        
        std::vector<arma::mat> hidden_layer_input;
        std::vector<arma::mat> hidden_layer_output;
        arma::mat final_hidden_layer_output;

        arma::mat X;
        arma::mat one_hot(int y, int n=10);
};
