#ifndef APP_H
#define APP_H

#include "model.h"
#include <vector>
#include <string>

/**
 * \file
 * \brief Header file for App class
 */

/**
 * @class App
 * @brief Provides a command-line interface for training and using a Perceptron model.
 *
 * The App class wraps a default and current Perceptron instance, offering methods
 * to predict, train, save/load models, and manage model configurations.
 */
class App {
public:
    /**
     * @brief Constructs the App with default error output enabled.
     *
     * Initializes both the default and current models by loading from the predefined file.
     */
    App();

    /**
     * @brief Constructs the App with specified error output setting.
     *
     * @param cur_error_output If true, error messages will be printed to stdout.
     */
    App(bool cur_error_output);

    /**
     * @brief Predicts the class of an image given command arguments.
     *
     * Expected args: ["predict", <image_path>]
     * @param args Vector of command arguments
     * @return 0 on successful prediction, -1 on error
     */
    int do_predict(const std::vector<std::string> args);

    /**
     * @brief Trains the current model using labeled data file.
     *
     * Expected args: ["train", <file_path>, <epochs>, <learning_rate>]
     * @param args Vector of command arguments
     * @return 0 on success, -1 on failure
     */
    int do_train(const std::vector<std::string> args);

    /**
     * @brief Creates a custom Perceptron model with specified hidden layer sizes.
     *
     * Expected args: ["make_custom_model", <size1>, <size2>, ...]
     * @param args Vector of command arguments specifying hidden layers
     * @return 0 on success, -1 on invalid input
     */
    int do_make_custom_model(const std::vector<std::string> args);

    /**
     * @brief Loads a Perceptron model from a file.
     *
     * Expected args: ["load_custom_model", <model_path>]
     * @param args Vector of command arguments
     * @return 0 on success, -1 if file not found or invalid
     */
    int do_load_custom_model(const std::vector<std::string> args);

    /**
     * @brief Saves the current model to a file.
     *
     * Expected args: ["save_model", <output_path>]
     * @param args Vector of command arguments
     * @return 0 on success, -1 on error
     */
    int do_save_model(const std::vector<std::string> args);

    /**
     * @brief Reloads the default Perceptron model from its source file.
     *
     * Restores both current and default models to initial state.
     */
    void do_load_default_model();

    /**
     * @brief Resets current training by copying the default model.
     */
    void reset_training();

    Perceptron defaultModel;  ///< The default pre-loaded Perceptron model
    Perceptron nn;            ///< The current Perceptron model in use

    bool error_output = true; ///< Enable printing of error messages

private:
    /**
     * @brief Checks existence of a file path.
     * @param path File path to check
     * @return True if file exists and is accessible, false otherwise
     */
    bool file_exists(const std::string& path);
};

#endif // APP_H
