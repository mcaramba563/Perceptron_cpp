#ifndef APP_H
#define APP_H

#include "model.h"
#include <vector>
#include <string>

class App {
public:
    App();

    int do_predict(const std::vector<std::string> args);
    int do_train(const std::vector<std::string> args);
    int do_make_custom_model(const std::vector<std::string> args);
    int do_load_custom_model(const std::vector<std::string> args);
    int do_save_model(const std::vector<std::string> args);
    void do_load_default_model();
    void reset_training();

private:
    Perceptron defaultModel;
    Perceptron nn;
    bool file_exists(std::string const& path);
};

#endif
