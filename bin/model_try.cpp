#include "app.h"
#include <sstream>
#include <string>

int main() {
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
        if (args.size() == 0) {
            continue;
        }
        if (args[0] == "exit")
            break;
        if (args[0] == "predict") {
            std::cout << app.do_predict(args) << "\n";
            continue;
        } else if (args[0] == "train") {
            app.do_train(args);
            continue;
        } else if (args[0] == "load_default_model") {
            app.do_load_default_model();
            continue;
        } else if (args[0] == "make_custom_model") {
            app.do_make_custom_model(args);
            continue;
        } else if (args[0] == "reset_training") {
            app.reset_training();
            continue;
        } else if (args[0] == "load_custom_model") {
            app.do_load_custom_model(args);
            continue;
        } else if (args[0] == "save_model") {
            app.do_save_model(args);
            continue;
        } else {
            std::cout << "Invalid mode. Available mods: exit, predict, train, load_default_model, make_custom_model, reset_training, load_custom_model, save_model\n";
        }
    }
    return 0;
}
