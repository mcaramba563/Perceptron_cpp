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

TEST_CASE("Test forward propagation") {
    Perceptron perceptron;
    arma::mat X = arma::randu<arma::mat>(1, 28*28);

    arma::mat output = perceptron.forward(X);
    CHECK(output.n_cols == perceptron.get_output_size());
    CHECK(abs(arma::as_scalar(arma::sum(output, 1)) - 1.0) < 1e-6);
}

TEST_CASE("Test backpropagation") {
    Perceptron perceptron;
    arma::mat X = arma::randu<arma::mat>(1, perceptron.get_input_size());

    perceptron.forward(X);
    perceptron.backprop(1);
}

TEST_CASE("Test training function") {
    Perceptron perceptron;
    std::vector<arma::mat> X;
    for (int i = 0;i < 5;i++)
        X.push_back(arma::randu<arma::mat>(1, perceptron.get_input_size()));
    std::vector<int> y{0, 1, 2, 3, 4};
    perceptron.train(X, y, 1, 0.01);
}

TEST_CASE("Test prediction function") {
    Perceptron perceptron;
    arma::mat X = arma::randu<arma::mat>(1, perceptron.get_input_size());

    int prediction = perceptron.predict(X);
    CHECK(prediction >= 0);
    CHECK(prediction < perceptron.get_output_size());
}

TEST_CASE("Test save and load model") {
    Perceptron perceptron;
    std::string model_path = "../../models/tmp_model";

    perceptron.save_model(model_path);
    CHECK(std::ifstream(model_path).good());

    Perceptron new_perceptron;
    new_perceptron.load_model(model_path);

    const auto original_weights = perceptron.get_weights_input_hidden();
    const auto loaded_weights = new_perceptron.get_weights_input_hidden();

    for (size_t i = 0; i < original_weights.size(); ++i) {
        CHECK(arma::approx_equal(original_weights[i], loaded_weights[i], "absdiff", 1e-4));
    }

    CHECK(arma::approx_equal(perceptron.get_weights_output_hidden(), new_perceptron.get_weights_output_hidden(), "absdiff", 1e-4));

    const auto& original_bias = perceptron.get_bias_input_hidden();
    const auto& loaded_bias = new_perceptron.get_bias_input_hidden();

    CHECK(original_bias.size() == loaded_bias.size());
    for (size_t i = 0; i < original_bias.size(); ++i) {
        CHECK(arma::approx_equal(original_bias[i], loaded_bias[i], "absdiff", 1e-4));
    }
}

