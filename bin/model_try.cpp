#include "app.h"
#include <sstream>
#include <string>

int main() {
    // predict ../../images/mnist_png/train/1/11107.png
    App app;
    while (true) {
        std::string input;
        std::getline(std::cin, input);
        
        std::vector<std::string> args;
        std::istringstream istream(input);

        std::string cur_arg;
        while (istream >> cur_arg) {
            args.push_back(cur_arg);
        }

        if (args[0] == "exit")
            break;
        if (args[0] == "predict") {
            app.doPredict(args);
            continue;
        } else if (args[0] == "train") {
            app.doTrain(args);
            continue;
        } else if (args[0] == "load_default_model") {
            app.doLoadDefaultModel();
            continue;
        } else if (args[0] == "make_custom_model") {
            app.doMakeCustomModel(args);
            continue;
        } else if (args[0] == "reset_training") {
            app.resetTraining();
            continue;
        } else if (args[0] == "load_custom_model") {
            app.doLoadCustomModel(args);
            continue;
        } else if (args[0] == "save_model") {
            app.doSaveModel(args);
            continue;
        } else {
            std::cout << "Invalid mode. Available mods: exit, predict, train, load_default_model, make_custom_model, reset_training, load_custom_model, save_model\n";
        }
    }
    return 0;
}
