//
// Created by sergio on 13/05/19.
//

#ifndef CPPFLOW_TENSOR_H
#define CPPFLOW_TENSOR_H

#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <tensorflow/c/c_api.h>
#include "Model.h"

class Model;

class Tensor {
public:
    Tensor(const Model& model, const std::string& operation);

    // Rule of five, moving is easy as the pointers can be copied, copying not as i have no idea how to copy
    // the contents of the pointer (i guess dereferencing won't do a deep copy)
    Tensor(const Tensor &tensor) = delete;
    Tensor(Tensor &&tensor) = default;
    Tensor& operator=(const Tensor &tensor) = delete;
    Tensor& operator=(Tensor &&tensor) = default;

    ~Tensor();

    void clean();

    template<typename T>
    void set_data(std::vector<T> new_data);

    template<typename T>
    void set_data(std::vector<T> new_data, const std::vector<int64_t>& new_shape);

    template<typename T>
    std::vector<T> get_data();

	std::vector<int64_t> get_shape();

private:
    TF_Tensor* val;
    TF_Output op;
    TF_DataType type;
    std::vector<int64_t> shape;
    std::unique_ptr<std::vector<int64_t>> actual_shape;
    void* data;
    int flag;

    // Aux functions
    void error_check(bool condition, const std::string& error);

    template <typename T>
    static TF_DataType deduce_type();

    void deduce_shape();

public:
    friend class Model;
};

#endif //CPPFLOW_TENSOR_H
