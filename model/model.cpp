// add predict_image
#include "model.h"

Perceptron::Perceptron(int input_size, std::vector<int> hidden_layers_size, int output_size, double learning_rate, int epochs)
    : input_size(input_size), hidden_layers_size(hidden_layers_size), output_size(output_size), learning_rate(learning_rate), epochs(epochs) {
    
    arma::arma_rng::set_seed(42);
    gen_weights();
    bias_gen();
}

void Perceptron::gen_weights() {
    weights_input_hidden.clear();
    weights_input_hidden.push_back(arma::randu<arma::mat>(input_size, hidden_layers_size[0]) - 0.5);

    for (size_t j = 1; j < hidden_layers_size.size(); ++j) {
        weights_input_hidden.push_back(arma::randu<arma::mat>(hidden_layers_size[j - 1], hidden_layers_size[j]) - 0.5);
    }

    weights_output_hidden = arma::randu<arma::mat>(hidden_layers_size.back(), output_size) - 0.5;
}

void Perceptron::bias_gen() {
    bias_input_hidden.clear();
    for (size_t j = 0; j < hidden_layers_size.size(); ++j) {
        bias_input_hidden.push_back(arma::randu<arma::mat>(1, hidden_layers_size[j]) - 0.5);
    }
}

arma::mat Perceptron::tanh_activation(const arma::mat& x) {
    return arma::tanh(x);
}

arma::mat Perceptron::tanh_derivative(const arma::mat& x) {
    return 1 - arma::square(arma::tanh(x));
}

arma::mat Perceptron::softmax(const arma::mat& x) {
    arma::mat exp_x = arma::exp(x - as_scalar(arma::max(x, 1)));
    // std::cout << x << " debug first\n";
    // std::cout << arma::max(x, 1) << " debug second\n";
    return exp_x / as_scalar(arma::sum(exp_x, 1));
}

arma::mat Perceptron::forward(const arma::mat& X) {
    this->X = X;
    
    hidden_layer_input.clear();
    hidden_layer_output.clear();
    
    for (size_t j = 0; j < weights_input_hidden.size(); ++j) {
        if (j == 0) {
            hidden_layer_input.push_back(X * weights_input_hidden[j] + bias_input_hidden[j]);
        } else {
            hidden_layer_input.push_back(hidden_layer_output[j - 1] * weights_input_hidden[j] + bias_input_hidden[j]);
        }
        hidden_layer_output.push_back(tanh_activation(hidden_layer_input[j]));
    }

    final_hidden_layer_output = tanh_activation(hidden_layer_output.back() * weights_output_hidden);
    // return final_hidden_layer_output;
    return softmax(final_hidden_layer_output);
}

void Perceptron::backprop(const int y_train) {
    arma::mat y_train_one_hot = one_hot(y_train, output_size);
    arma::mat error_final = y_train_one_hot - final_hidden_layer_output;
    arma::mat d_final_input = error_final % tanh_derivative(final_hidden_layer_output);

    std::vector<arma::mat> d_hidden_layers(weights_input_hidden.size());
    
    for (int j = weights_input_hidden.size() - 1; j >= 0; --j) {
        if (j == weights_input_hidden.size() - 1) {
            d_hidden_layers[j] = (d_final_input * weights_output_hidden.t()) % tanh_derivative(hidden_layer_output[j]);
        } else {
            d_hidden_layers[j] = (d_hidden_layers[j + 1] * weights_input_hidden[j + 1].t()) % tanh_derivative(hidden_layer_output[j]);
        }
    }


    assert(hidden_layer_output.back().n_rows == 1 && d_final_input.n_rows == 1);
    weights_output_hidden += hidden_layer_output.back().t() * d_final_input * learning_rate;
    
    for (size_t j = 0; j < weights_input_hidden.size(); ++j) {
        if (j > 0) {
            assert(hidden_layer_output[j - 1].n_rows == 1 && d_hidden_layers[j].n_rows == 1);
            weights_input_hidden[j] += hidden_layer_output[j - 1].t() * d_hidden_layers[j] * learning_rate;
            bias_input_hidden[j] += d_hidden_layers[j] * learning_rate;
        } else {
            weights_input_hidden[j] += X.t() * d_hidden_layers[j] * learning_rate;
            bias_input_hidden[j] += d_hidden_layers[j] * learning_rate;
        }
    }
}

void Perceptron::train(const arma::mat& X_train, const arma::vec& y_train, int epochs, double learning_rate) {
    this->epochs = epochs;
    this->learning_rate = learning_rate;
    // arma::mat y_train_one_hot = arma::zeros(y_train.n_rows, output_size);
    // for (size_t i = 0; i < y_train.n_rows; ++i) {
    //     y_train_one_hot(i, y_train(i)) = 1;
    // }
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t ind = 0; ind < X_train.n_rows; ++ind) {
            forward(X_train.row(ind));
            backprop(y_train(ind));
        }
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " completed" << std::endl;
    }
}

int Perceptron::predict(const arma::mat& X) {
    arma::mat predicted_output = forward(X);
    return predicted_output.index_max();
}

void Perceptron::save_model(const std::string& path) {
    arma::mat save_array;
    save_array.insert_rows(0, arma::vectorise(weights_output_hidden));
    for (const auto& w : weights_input_hidden) {
        save_array.insert_rows(save_array.n_rows, arma::vectorise(w));
    }
    save_array.save(path, arma::arma_binary);
}

void Perceptron::load_model(const std::string& path) {
    arma::mat save_array;
    save_array.load(path, arma::arma_binary);
    weights_output_hidden = arma::reshape(save_array.rows(0, output_size - 1), output_size, hidden_layers_size.back());
    int row_start = output_size;
    for (size_t j = 0; j < weights_input_hidden.size(); ++j) {
        weights_input_hidden[j] = arma::reshape(save_array.rows(row_start, row_start + hidden_layers_size[j] - 1), hidden_layers_size[j], hidden_layers_size[j]);
        row_start += hidden_layers_size[j];
    }
}

arma::mat Perceptron::one_hot(int y, int n) {
    arma::mat mat_one_hot(1, n, arma::fill::zeros);
    mat_one_hot(0, y) = 1.0;
    return mat_one_hot;
}