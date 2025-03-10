#include "model.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

TEST_CASE("Test weight initialization") {
    Perceptron perceptron;
    perceptron.gen_weights();
    perceptron.bias_gen();

    const auto weights_input_hidden = perceptron.get_weights_input_hidden();
    const auto hidden_layer_input = perceptron.get_hidden_layer_input();

    const int input_size = perceptron.get_input_size();
    const int output_size = perceptron.get_output_size();
    const auto hidden_layers_size = perceptron.get_hidden_layers_size();
    CHECK(weights_input_hidden.size() == hidden_layers_size.size());
    for (size_t i = 0;i < weights_input_hidden.size();i++) {
        if (i == 0) {
            CHECK(weights_input_hidden[i].n_rows == input_size);
            CHECK(weights_input_hidden[i].n_cols == hidden_layers_size[0]);
        } else {
            CHECK(weights_input_hidden[i].n_rows == hidden_layers_size[i - 1]);
            CHECK(weights_input_hidden[i].n_cols == hidden_layers_size[i]);
        }
    }

    CHECK(perceptron.get_weights_output_hidden().n_rows == hidden_layers_size.back());
    CHECK(perceptron.get_weights_output_hidden().n_cols == output_size);
}

TEST_CASE("Test bias initialization") {
    Perceptron perceptron;
    perceptron.bias_gen();

    const auto bias_input_hidden = perceptron.get_bias_input_hidden();
    const auto hidden_layers_size = perceptron.get_hidden_layers_size();

    CHECK(bias_input_hidden.size() == hidden_layers_size.size());

    for (size_t i = 0; i < bias_input_hidden.size(); ++i) {
        CHECK(bias_input_hidden[i].n_rows == 1);
        CHECK(bias_input_hidden[i].n_cols == hidden_layers_size[i]);
    }
}

TEST_CASE("Test tanh activation function") {
    Perceptron perceptron;
    arma::vec x = {-1, 0, 1};
    arma::vec expected = arma::tanh(x);
    
    CHECK(arma::approx_equal(perceptron.tanh_activation(x), expected, "absdiff", 1e-6));
}

TEST_CASE("Test tanh derivative") {
    Perceptron perceptron;
    arma::vec x = {-1, 0, 1};
    arma::vec expected = 1 - arma::square(arma::tanh(x));

    CHECK(arma::approx_equal(perceptron.tanh_derivative(x), expected, "absdiff", 1e-6));
}

TEST_CASE("Test softmax function") {
    Perceptron perceptron;
    arma::mat x = {1, 2, 3};
    arma::mat expected = {0.0900, 0.2447, 0.6652};
    CHECK(arma::approx_equal(perceptron.softmax(x), expected, "absdiff", 1e-4));
}
/*
TEST_CASE("Test forward propagation") {
    Perceptron perceptron;
    arma::vec X = arma::randu(perceptron.input_size);

    arma::vec output = perceptron.forward(X);
    CHECK(output.n_rows == perceptron.output_size);
    CHECK(arma::approx_equal(arma::sum(output), arma::vec{1.0}, "absdiff", 1e-6));
}

TEST_CASE("Test backpropagation") {
    Perceptron perceptron;
    arma::vec X = arma::randu(perceptron.input_size);
    arma::vec y = arma::zeros(perceptron.output_size);
    y(0) = 1;

    perceptron.forward(X);
    perceptron.backprop(y);
}

TEST_CASE("Test training function") {
    Perceptron perceptron;
    arma::mat X = arma::randu(10, perceptron.input_size);
    arma::ivec y = arma::randi<arma::ivec>(10, arma::distr_param(0, perceptron.output_size - 1));

    perceptron.train(X, y, 1, 0.01);
}

TEST_CASE("Test prediction function") {
    Perceptron perceptron;
    arma::vec X = arma::randu(perceptron.input_size);

    int prediction = perceptron.predict(X);
    CHECK(prediction >= 0);
    CHECK(prediction < perceptron.output_size);
}

TEST_CASE("Test save and load model") {
    Perceptron perceptron;
    std::string model_path = "../../models/tmp_model.bin";

    perceptron.save_model(model_path);
    CHECK(std::ifstream(model_path).good());

    Perceptron new_perceptron;
    new_perceptron.load_model(model_path);

    const auto original_weights = perceptron.get_weights_input_hidden();
    const auto loaded_weights = new_perceptron.get_weights_input_hidden();

    CHECK(original_weights.size() == loaded_weights.size());
    for (size_t i = 0; i < original_weights.size(); ++i) {
        CHECK(arma::approx_equal(original_weights[i], loaded_weights[i], "absdiff", 1e-6));
    }

    CHECK(arma::approx_equal(perceptron.get_weights_output_hidden(), new_perceptron.get_weights_output_hidden(), "absdiff", 1e-6));

    const auto& original_bias = perceptron.get_bias_input_hidden();
    const auto& loaded_bias = new_perceptron.get_bias_input_hidden();

    CHECK(original_bias.size() == loaded_bias.size());
    for (size_t i = 0; i < original_bias.size(); ++i) {
        CHECK(arma::approx_equal(original_bias[i], loaded_bias[i], "absdiff", 1e-6));
    }
}
*/
