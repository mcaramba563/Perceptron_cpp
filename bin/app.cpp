#include "app.h"
#include <iostream>
#include <fstream>
#include <sstream>

App::App() {
    nn = Perceptron();
    defaultModel = Perceptron();
    defaultModel.load_model("../../models/default_model");
    nn.load_model("../../models/default_model");
}

int App::doPredict(const std::vector<std::string> args) {
    if (args.size() < 2) {
        std::cerr << "You need to specify the path to file" << std::endl;
        return -1;
    }

    int ans = nn.predict_image(args[1]);
    if (ans == -1) {
        std::cerr << "Something went wrong. Check the path to the file and try again." << std::endl;
        return -1;
    }

    std::cout << ans << std::endl;
    return 0;
}

int App::doTrain(const std::vector<std::string> args) {
    if (args.size() < 4) {
        std::cerr << "Specify the path to file, epochs, and learning rate" << std::endl;
        return -1;
    }

    int epochs = std::stoi(args[2]);
    double learningRate = std::stod(args[3]);
    if (epochs <= 0 || learningRate <= 0) {
        std::cerr << "Check epochs and learning_rate values" << std::endl;
        return -1;
    }

    std::vector<std::string> paths;
    std::vector<int> labels;
    std::string filename = args[1];
    if (!fileExists(filename)) {
        std::cerr << "Check the file name" << std::endl;
        return -1;
    }
    
    // nn.train_on_specific_images(filename);
    nn.train_on_specific_images(filename, epochs, learningRate);
    return 0;
}

bool App::fileExists(std::string const& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

int App::doMakeCustomModel(const std::vector<std::string> args) {
    if (args.size() < 2) {
        std::cerr << "Specify the count of neurons in each layer" << std::endl;
        return -1;
    }

    std::vector<int> hiddenLayersSize;
    try {
        std::istringstream istream(args[1]);
        std::string cur_number;
        while (istream >> cur_number) {
            hiddenLayersSize.push_back(std::stoi(cur_number));
        }
    } catch (const std::exception& e) {
        std::cerr << "Invalid neuron count: " << e.what() << std::endl;
        return -1;
    }

    nn = Perceptron(28 * 28, hiddenLayersSize, 10, 0.01, 1);
    defaultModel = nn; 
    return 0;
}

int App::doLoadCustomModel(const std::vector<std::string> args) {
    if (args.size() < 2) {
        std::cerr << "Specify the path to the file" << std::endl;
        return -1;
    }

    if (!fileExists(args[1])) {
        std::cerr << "Invalid file name" << std::endl;
        return -1;
    }

    nn.load_model(args[1]);
    defaultModel = nn;
    return 0;
}

int App::doSaveModel(const std::vector<std::string> args) {
    if (args.size() < 2) {
        std::cerr << "Specify the path to the file" << std::endl;
        return -1;
    }

    try {
        nn.save_model(args[1]);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
        return -1;
    }
}

void App::doLoadDefaultModel() {
    nn.load_model("../../models/default_model");
    defaultModel.load_model("../../models/default_model");
}

void App::resetTraining() {
    nn = defaultModel;
}
