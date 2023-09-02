#include "../../include/Model.h"
#include "../../include/Tensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <numeric>
#include <iomanip>

namespace py = pybind11;

class Singleton {
private:
    static bool instanceFlag;
    static Singleton *single;
    Model model;
    Tensor input, output;

    Singleton(const std::string &fname, const std::string &inputName,
              const std::string &outputName) :
            model{fname}, input{model, inputName}, output{model, outputName} {
    }

public:
    static Singleton *getInstance();

    static Singleton *getInstance(const std::string &fname, const std::string &inputName,
                                  const std::string &outputName);

    std::vector<float> run_model(std::vector<float> input_data,  std::vector<int64_t> input_data_shape);

    ~Singleton() {
        instanceFlag = false;
    }
};

bool Singleton::instanceFlag = false;
Singleton *Singleton::single = NULL;

Singleton *Singleton::getInstance(const std::string &fname, const std::string &inputName,
                                  const std::string &outputName) {
    if (!instanceFlag) {
        single = new Singleton(fname, inputName, outputName);
        instanceFlag = true;
        return single;
    } else {
        return single;
    }
}

Singleton *Singleton::getInstance() {
    return single;
}

std::vector<float> Singleton::run_model(std::vector<float> input_data, std::vector<int64_t> input_data_shape) {
    input.set_data(input_data, input_data_shape);
    model.run({&input}, output);
    std::vector<float> res = output.get_data<float>();
    return res;
}



void load_model(const std::string &fname, const std::string &inputName,
                const std::string &outputName) {
    Singleton *sc1;
    sc1 = Singleton::getInstance(fname, inputName, outputName);
}

std::vector<float> run_model(std::vector<float> input_data, std::vector<int64_t> input_data_shape) {
    Singleton *sc2;
    sc2 = Singleton::getInstance();
    return sc2->run_model(input_data, input_data_shape);
}


PYBIND11_MODULE(tf_c_inference, m) {
    m.doc() = R"pbdoc(
        Pybnind module to invoke Tensorflow C infernce
        -----------------------

        .. currentmodule:: tf_c_inference

        .. autosummary::
           :toctree: _generate

           load_model
           run_model
    )pbdoc";


    //m.def("run1", &run1, "Run inference");
    m.def("load_model", &load_model, "Load Model");
    m.def("run_model", &run_model, "Run inference");


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}



