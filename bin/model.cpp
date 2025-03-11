#include "model.h"
#include <stdexcept>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif
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
    // std::cout << input_size << " ";
    // for (int cur : hidden_layers_size)
    //     std::cout << cur << " ";
    // std::cout << output_size << "\n";

   
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

void Perceptron::train(const std::vector<arma::mat>& X_train, const std::vector<int>& y_train, int epochs, double learning_rate) {
    this->epochs = epochs;
    this->learning_rate = learning_rate;
    std::cout << "Starting training\n";
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t ind = 0; ind < X_train.size(); ++ind) {
            forward(X_train[ind]);
            backprop(y_train[ind]);
        }
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " completed" << std::endl;
    }
}

int Perceptron::predict(const arma::mat& X) {
    arma::mat predicted_output = forward(X);
    return predicted_output.index_max();
}

int Perceptron::predict_image(std::string path) {
    return predict(read_image(path));
}

void Perceptron::save_model(const std::string& path) {
    arma::mat save_array;

    arma::mat model_structure(1, hidden_layers_size.size() + 3);
    model_structure(0, 0) = hidden_layers_size.size();
    model_structure(0, 1) = input_size;
    for (size_t i = 0; i < hidden_layers_size.size(); ++i) {
        model_structure(0,  i + 2) = hidden_layers_size[i];
    }
    model_structure(0, hidden_layers_size.size() + 2) = output_size;
    // std::cout << model_structure << "\n\n\n";
    save_array.insert_rows(0, arma::vectorise(model_structure)); 

    save_array.insert_rows(save_array.n_rows, arma::vectorise(weights_output_hidden));

    for (const auto& w : weights_input_hidden) {
        save_array.insert_rows(save_array.n_rows, arma::vectorise(w));
    }
    
    for (const auto& b : bias_input_hidden) {
        save_array.insert_rows(save_array.n_rows, arma::vectorise(b));
    }

    save_array.save(path, arma::arma_binary);
}


void Perceptron::load_model(const std::string& path) {
    arma::mat save_array;
    save_array.load(path, arma::arma_binary);

    int offset = 0;
    int cur_hidden_layers_size = static_cast<int>(save_array(0, 0));
    offset++;

    arma::mat model_structure = save_array.rows(offset, offset + cur_hidden_layers_size + 1);
    model_structure.reshape(1, cur_hidden_layers_size + 2);
    offset += cur_hidden_layers_size + 2;
    // std::cout << model_structure << "\n\n\n";
    input_size = model_structure(0, 0);
    hidden_layers_size.clear();
    for (size_t i = 1; i < model_structure.n_cols - 1; ++i) {
        hidden_layers_size.push_back(static_cast<int>(model_structure(0, i)));
    }
    output_size = static_cast<int>(model_structure(0, model_structure.n_cols - 1));

    gen_weights();
    bias_gen();

    int output_cols = output_size;
    int output_rows = hidden_layers_size.back();
    weights_output_hidden.set_size(output_rows, output_cols);
    weights_output_hidden = arma::reshape(save_array.rows(offset, offset + (output_rows * output_cols) - 1),
                                          output_rows, output_cols);
    offset += output_rows * output_cols;

    for (size_t j = 0; j < hidden_layers_size.size(); ++j) {
        int rows = (j == 0) ? input_size : hidden_layers_size[j - 1];
        int cols = hidden_layers_size[j];

        weights_input_hidden[j].set_size(rows, cols);
        weights_input_hidden[j] = arma::reshape(save_array.rows(offset, offset + (rows * cols) - 1), rows, cols);
        offset += rows * cols;
    }

    for (size_t j = 0; j < bias_input_hidden.size(); ++j) {
        int cols = bias_input_hidden[j].n_cols;

        bias_input_hidden[j] = arma::reshape(save_array.rows(offset, offset + cols - 1), 1, cols);
        offset += cols;
    }

}




arma::mat Perceptron::one_hot(int y, int n) {
    arma::mat mat_one_hot(1, n, arma::fill::zeros);
    mat_one_hot(0, y) = 1.0;
    return mat_one_hot;
}

bool Perceptron::image_path_correct(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

arma::mat Perceptron::read_image(const std::string& path, int image_bbp) {
    int width, height, bpp;
    if (!image_path_correct(path)) {
        std::cout << "Invalid image path: " << path << "\n";
        throw std::runtime_error{"loading image error"};
    }    
    uint8_t* image = stbi_load(path.c_str(), &width, &height, &bpp, image_bbp);
    if (image_bbp != bpp) {
        std::cout << "Error: image bpp does not match with input bpp, image path: " << path << "\n";
        throw std::runtime_error{"imvalid bpp"};
    }
    
    std::vector<double> image_vec(height * width, 0.0);
    for (int i = 0;i < height * width;i++) {
        image_vec[i] = (double)image[i] / 255.0;
    }
    
    arma::mat ans = arma::conv_to<arma::mat>::from(image_vec);
    ans.reshape(1, width * height);
    return ans;
}


std::vector<arma::mat> Perceptron::read_images(const std::vector<std::string>& pathes) {
    std::vector<arma::mat> ans;
    for (const std::string& cur_path : pathes) {
        ans.push_back(read_image(cur_path, 1));
    }
    return ans;
}


void Perceptron::train_on_specific_images(std::string path, int cur_epochs, double cur_learning_rate) {
    if (cur_epochs != -100){
        this->epochs = cur_epochs;
    }
    if (learning_rate != -100) {
        this->learning_rate = cur_learning_rate;
    }
    std::ifstream fin(path);
    int n;
    fin >> n;
    std::vector<std::string> pathes(n);
    std::vector<int> labels(n);
    for (int i = 0;i < n;i++) {
        fin >> pathes[i] >> labels[i];
    }
    std::vector<arma::mat> images = read_images(pathes);
    train(images, labels, epochs, learning_rate);
}
