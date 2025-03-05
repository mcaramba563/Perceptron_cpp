#include "model.h"



int main() {
    Perceptron p;
    // arma::mat cur = arma::randu<arma::mat>(1, 28*28);
    
    // arma::mat forward_cur = p.forward(cur);
    // std::cout << forward_cur << "\n";
    // for (int i = 0;i < 300;i++)
    //     p.backprop(2);

    // forward_cur = p.forward(cur);
    // std::cout << forward_cur << "\n";
    // std::cout << "------------------------------------------------------\n";
    // arma::mat cur_1(1, 28 * 28, arma::fill::ones);
    // arma::mat forward_cur_1 = p.forward(cur);
    // std::cout << forward_cur_1 << "\n";
    // for (int i = 0;i < 300;i++)
    //     p.backprop(1);

    // forward_cur_1 = p.forward(cur_1);
    // std::cout << forward_cur_1 << "\n";
    arma::mat cur = p.read_image("../../test_libraries/image_read_test/21.png");
    arma::mat forward_cur = p.forward(cur);
    std::cout << forward_cur << "\n";
    for (int i = 0;i < 300;i++)
        p.backprop(2);

    forward_cur = p.forward(cur);
    std::cout << forward_cur << "\n";
    return 0;
}