#ifndef _DSL_HPP
#define _DSL_HPP

#include "matrice.hpp"
#include "time_manager.hpp"
#include <pthread.h>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <utility>
#include <memory>


// Activations scalaires et derivees
inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double sigmoid_derivative(double s) { return s * (1.0 - s); }
inline double tanh_derivative(double t) { return 1.0 - t * t; }

class LossFunction {
public:
    virtual ~LossFunction() = default;
    virtual double forward(const Matrice& logits, const std::vector<int>& targets) = 0;
    virtual Matrice backward() = 0;
    virtual LossFunction* clone() const = 0;
};

typedef struct {
    int index;
    int debut;
    int fin;
    double alpha;
    double cout;        
    int batch_size;       
    const Tensor* X;             
    const std::vector<int>* Y;   
    const int* ordre;
    LossFunction* loss;   
    const std::vector<Tensor>* W_conv_init;
    const std::vector<std::vector<double>>* b_conv_init;
    const std::vector<Matrice>* W_dense_init;
    const std::vector<std::vector<double>>* b_dense_init;
    std::vector<Tensor> W_conv_local;    
    std::vector<std::vector<double>> b_conv_local; 
    std::vector<Matrice> W_dense_local;   
    std::vector<std::vector<double>> b_dense_local; 
} Arg_CNN;

typedef struct {
    int index;                 
    int debut;                 
    int fin;                  
    int batch_size;           
    double alpha;              
    double cout;               
    const int* ordre;          
    const Sequency* X_seq;     
    int target_col;
    const Matrice* W_ih_init;  
    const Matrice* b_ih_init;  
    const Matrice* W_hy_init;
    const Matrice* b_y_init;
    Matrice dW_ih_local;      
    Matrice db_ih_local;       
    Matrice dW_hy_local;
    Matrice db_y_local;
    int input_size;            
    int hidden_size;           
    int seq_len;               
} Arg_LSTM;

typedef struct {
    std::vector<Arg_CNN> threads_results; 
    Matrice erreurs_threads;
    std::vector<Tensor> final_W_conv;
    std::vector<std::vector<double>> final_b_conv;
    std::vector<Matrice> final_W_dense;
    std::vector<std::vector<double>> final_b_dense;
} Par_CNN_thread;

typedef struct {
    double cout_final;
    int epoques_effectuees;
    double temps_execution;
    Matrice W_ih;
    Matrice b_ih;
    Matrice W_hy;
    Matrice b_y;
} Par_LSTM_thread;

int* range(int size);
int select_random_number(int min, int max);
void shuffle(int* array, int n);

Matrice sigmoid(Matrice A);
Matrice softmax(Matrice A);

Tensor create_tensor(int n, int c, int h, int w, double fill = 0.0);
Tensor pad_tensor(const Tensor& x, int pad);

class Conv2D {
public:
    int in_channels, out_channels, kernel_size, stride, padding;
    Tensor weights;
    std::vector<double> biases;
    Tensor dweights;
    std::vector<double> dbiases;
    Tensor x_cached;

    Conv2D(int in_c, int out_c, int k_size, int str = 1, int pad = 0);
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& dout, double lr);
};

class MaxPool2D {
public:
    int pool_size, stride;
    Tensor x_cached;
    std::map<std::tuple<int, int, int, int>, std::pair<int, int>> max_indices;

    MaxPool2D(int p_size = 2, int str = 2);
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& dout, double lr);
};

class Flatten {
public:
    std::vector<int> orig_shape;
    Matrice forward(const Tensor& x);
    Tensor backward(const Matrice& dout, double lr);
};

class Dense {
public:
    int in_features, out_features;
    Matrice weights;
    std::vector<double> biases;
    Matrice x_cached;

    Dense(int in_f, int out_f);
    Matrice forward(const Matrice& x);
    Matrice backward(const Matrice& dout, double lr);
};

class ReLU {
public:
    Matrice x_cached;
    Matrice forward(const Matrice& x);
    Matrice backward(const Matrice& dout, double lr);
};

class CategoricalCrossEntropyWithSoftmax : public LossFunction {
public:
    Matrice logits_cached;
    std::vector<int> targets_cached;
    Matrice probs;

    double forward(const Matrice& logits, const std::vector<int>& targets) override;
    Matrice backward() override;
    LossFunction* clone() const override { return new CategoricalCrossEntropyWithSoftmax(*this); }
};

void* cnn_local(void* a);
Par_CNN_thread cnn_reduce(Tensor X, std::vector<int> Y, LossFunction& loss, double alpha, double seuil, int n_epoch, int n_threads, int batch_size = 32);

void* lstm_local(void* a);
double lstm_reduce(int n_threads, int input_size, int hidden_size, int seq_len, 
                   Matrice* W_ih_global, Matrice* b_ih_global, Matrice* W_hy_global, Matrice* b_y_global,
                   const Sequency& X_seq, int target_col, const std::vector<int>& ordre, double alpha);

#endif // _DSL_HPP
