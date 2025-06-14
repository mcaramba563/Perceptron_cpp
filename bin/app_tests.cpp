#include "app.h"
#include <stdexcept>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

TEST_CASE("Test valid predictions") {
    App app_instance(false);
    

    std::vector<std::pair<std::vector<std::string>, int>> test_cases = {
        {{"predict", "../../images/mnist_png/test/0/10.png"}, 0},
        {{"predict", "../../images/mnist_png/test/1/2.png"}, 1},
        {{"predict", "../../images/mnist_png/test/2/1.png"}, 2},
        {{"predict", "../../images/mnist_png/test/3/18.png"}, 3},
        {{"predict", "../../images/mnist_png/test/4/4.png"}, 4},
        {{"predict", "../../images/mnist_png/test/5/15.png"}, 5},
        {{"predict", "../../images/mnist_png/test/6/11.png"}, 6},
        {{"predict", "../../images/mnist_png/test/7/0.png"}, 7},
        {{"predict", "../../images/mnist_png/test/8/61.png"}, 8},
        {{"predict", "../../images/mnist_png/test/9/7.png"}, 9},
    };

    int correct_predictions = 0;
    for (const auto& [command, expected] : test_cases) {
        int actual = app_instance.do_predict(command);
        if (actual == expected) {
            correct_predictions++;
        }
    }

    CHECK(correct_predictions == 10);
}

TEST_CASE("Test incorrect predictions") {
    App app_instance(false);
    
    
    CHECK(app_instance.do_predict({"predict"}) == -1);
    CHECK(app_instance.do_predict({"predict", "../../images/mnist_png/test/9asdad/7.png"}) == -1);
}

TEST_CASE("Test incorrect training cases") {
    App app_instance(false);
    
    CHECK(app_instance.do_train({"train", "../test_train.txt"}) == -1);
    CHECK(app_instance.do_train({"train", "../test_train.txt", "2"}) == -1);
    CHECK(app_instance.do_train({"train", "../test_train.txt", "-1", "0.01"}) == -1);
    CHECK(app_instance.do_train({"train", "asd.txt", "1", "0.01"}) == -1);
    CHECK(app_instance.do_train({"train", "tests/file_with_incorrect_names.txt", "1", "0.01"}) == -1);
}


TEST_CASE("Test correct training") {
    App app_instance(false);
    
    CHECK(app_instance.do_train({"train", "../test_train.txt", "1", "0.01"}) == 0);
}

TEST_CASE("Test custom model creation and training") {
    App app_instance(false);
    
    CHECK(app_instance.do_make_custom_model({"make_custom_model", "400", "256", "128"}) == 0);
    CHECK(app_instance.do_train({"train", "../test_train.txt", "2", "0.01"}) == 0);

    std::vector<std::pair<std::vector<std::string>, int>> test_cases = {
        {{"predict", "../../images/mnist_png/test/0/10.png"}, 0},
        {{"predict", "../../images/mnist_png/test/1/2.png"}, 1},
        {{"predict", "../../images/mnist_png/test/2/1.png"}, 2},
        {{"predict", "../../images/mnist_png/test/3/18.png"}, 3},
        {{"predict", "../../images/mnist_png/test/4/4.png"}, 4},
        {{"predict", "../../images/mnist_png/test/5/15.png"}, 5},
        {{"predict", "../../images/mnist_png/test/6/11.png"}, 6},
        {{"predict", "../../images/mnist_png/test/7/0.png"}, 7},
        {{"predict", "../../images/mnist_png/test/8/61.png"}, 8},
        {{"predict", "../../images/mnist_png/test/9/7.png"}, 9},
    };

    int correct_predictions = 0;
    for (const auto& [command, expected] : test_cases) {
        if (app_instance.do_predict(command) == expected) {
            correct_predictions++;
        }
    }

    CHECK(correct_predictions >= 7);
}

TEST_CASE("Test saving and loading model") {
    App app_instance(false);
    
    CHECK(app_instance.do_save_model({"save_model", "../../models/default_model"}) == 0);
    CHECK(app_instance.do_load_custom_model({"load_custom_model", "../../models/default_model"}) == 0);
}

TEST_CASE("Test edge cases for training") {
    App app_instance(false);
    
    CHECK(app_instance.do_train({"train", "test_train.txt", "10", "-1", "0.01"}) == -1);
    CHECK(app_instance.do_train({"train", "test_train.txt", "10", "1", "-0.01"}) == -1);
    CHECK(app_instance.do_train({"train", "test_train.txt", "10", "1", "0"}) == -1);
}

TEST_CASE("Test loading model and making predictions") {
    App app_instance(false);
    
    app_instance.do_load_default_model();
    app_instance.do_save_model({"save_model", "../../models/tmp_model"});
    CHECK(app_instance.do_load_custom_model({"load_custom_model", "../../models/tmp_model"}) == 0);

    std::vector<std::pair<std::vector<std::string>, int>> test_cases = {
        {{"predict", "../../images/mnist_png/test/0/10.png"}, 0},
        {{"predict", "../../images/mnist_png/test/1/2.png"}, 1},
        {{"predict", "../../images/mnist_png/test/2/1.png"}, 2},
        {{"predict", "../../images/mnist_png/test/3/18.png"}, 3},
        {{"predict", "../../images/mnist_png/test/4/4.png"}, 4},
        {{"predict", "../../images/mnist_png/test/5/15.png"}, 5},
        {{"predict", "../../images/mnist_png/test/6/11.png"}, 6},
        {{"predict", "../../images/mnist_png/test/7/0.png"}, 7},
        {{"predict", "../../images/mnist_png/test/8/61.png"}, 8},
        {{"predict", "../../images/mnist_png/test/9/7.png"}, 9},
    };

    int correct_predictions = 0;
    for (const auto& [command, expected] : test_cases) {
        if (app_instance.do_predict(command) == expected) {
            correct_predictions++;
        }
    }

    CHECK(correct_predictions >= 8);
}

