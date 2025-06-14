# Perceptron C++ Neural Network Application

This repository contains a C++ implementation of a multilayer perceptron (MLP) using Armadillo for matrix operations and stb\_image for image loading. It supports training, prediction, and persistence (save/load) of models for image recognition tasks, such as digit classification on the MNIST dataset.

---

## Features

* **Custom Model Architectures**: Define arbitrary hidden layer sizes at runtime.
* **Train on Image Data**: Train the network on labeled datasets of images.
* **Randomized Training**: Sample random images for quick experiments.
* **Interactive CLI**: Command-line interface to issue training, prediction, and model management commands.
* **Persistence**: Save and load model weights and biases to/from disk.
* **Default Model**: Ship with a pretrained default model for immediate use.
* **Reset Functionality**: Revert to the default or initial model state.

---

## Prerequisites

* C++17 (or later)
* [Armadillo](https://arma.sourceforge.net/) (matrix computations)
* [stb\_image.h](https://github.com/nothings/stb) (image loading)
* A C++ build system (Make, CMake, etc.)

---

## Installation & Build

1. **Clone the repository**

   ```bash
   git clone https://github.com/mcaramba563/Perceptron_cpp.git
   cd Perceptron_cpp
   ```

2. **Ensure dependencies** are installed (Armadillo, stb\_image).

3. **Build** using CMake (example):

   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

4. This produces an executable (e.g., `perceptron_app`).

---

## Usage

Run the CLI application:

```bash
./perceptron_app
```

Once started, you can issue the following commands:

### 1. Create a Custom Model

```
make_custom_model <hidden1> <hidden2> ...
```

Example:

```
make_custom_model 400 256 128
```

### 2. Train on Custom Data

```
train <dataset_file> <epochs> <learning_rate>
```

* `dataset_file`

  * A text file listing `<image_path> <label>` per line.
    Example:

```
train data/train_list.txt 5 0.01
```

### 3. Make Predictions

```
predict <image_path>
```

Example:

```
predict images/test/0_10.png
```

### 4. Save the Model

```
save_model <output_path>
```

Example:

```
save_model models/my_model.bin
```

### 5. Load a Saved Model

```
load_custom_model <model_path>
```

Example:

```
load_custom_model models/my_model.bin
```

### 6. Load Default Model

```
load_default_model
```

### 7. Reset Training

```
reset_training
```

(Reverts to the state before any training.)

### 8. Exit

```
exit
```

---

## Example Workflow

```bash
./perceptron_app

make_custom_model 400 256 128
train train_list.txt 5 0.01

save_model models/tmp_model

load_default_model
predict images/test/7_12.png

exit
```

---
