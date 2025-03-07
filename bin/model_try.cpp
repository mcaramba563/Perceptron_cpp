#include "app.h"

int main() {
    // Perceptron p(28 * 28, {400, 256, 128}, 10, 0.01, 5);
    // p.save_model("../../models/tmp_model");
    // p.load_model("../../models/first_model_5_epochs");

    // // p.train_on_specific_images("../test_train.txt");
    // arma::mat image_0 = p.read_image("../../test_libraries/image_read_test/21.png");
    // arma::mat cur = p.forward(image_0);
    // std::cout << cur << "\n";
    // std::cout << "------------------------------------------------------\n";
    // std::cout << p.predict(image_0) << "\n";

    // p.save_model("../../models/first_model_5_epochs");
    // std::cout << "asd";
    App my_app;
    my_app.doMakeCustomModel({"", "400 200 128"});
    
    // my_app.doTrain({"", "../test_train.txt", "5", "0.01"});
    // my_app.doSaveModel({"", "../../models/tmp_model"});

    my_app.doPredict({"", "../../images/mnist_png/train/0/21.png"});
    my_app.doLoadDefaultModel();
    my_app.doPredict({"", "../../images/mnist_png/train/0/21.png"});
    return 0;
}