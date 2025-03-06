#include "model.h"



int main() {
    Perceptron p(28 * 28, {400, 256, 128}, 10, 0.01, 2);

    p.train_on_specific_images("../test_train.txt");
    arma::mat image_0 = p.read_image("../../test_libraries/image_read_test/21.png");
    arma::mat cur = p.forward(image_0);
    std::cout << cur << "\n";
    std::cout << "------------------------------------------------------\n";
    std::cout << p.predict(image_0) << "\n";

    return 0;
}