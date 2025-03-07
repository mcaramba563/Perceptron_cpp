#ifndef APP_H
#define APP_H

#include "model.h"
#include <vector>
#include <string>

class App {
public:
    App();

    int doPredict(const std::vector<std::string> args);
    int doTrain(const std::vector<std::string> args);
    int doMakeCustomModel(const std::vector<std::string> args);
    int doLoadCustomModel(const std::vector<std::string> args);
    int doSaveModel(const std::vector<std::string> args);
    void doLoadDefaultModel();
    void resetTraining();

private:
    Perceptron defaultModel;
    Perceptron nn;
};

#endif
