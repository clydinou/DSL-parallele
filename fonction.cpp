#include "fonction.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>

// Références aux variables globales de main.cpp
extern TimePoint w_time_par1, w_time_par2;
extern double work_time_par;

CNN_Weights_Par_Thread CNNAlgorithm(double alpha, double seuil, int n_epoch, std::string filename, int n_threads) {
    Tensor X, aux;
    Matrice Y_mat;

    aux.n = 10000;
    aux.c = 1;
    aux.h = 28;
    aux.w = 28;
    readTensor(&aux, &Y_mat, filename);
    X = aux;

    std::vector<int> Y(aux.n);
    for (int i = 0; i < aux.n; i++) {
        Y[i] = static_cast<int>(Y_mat.contenu[i][0]);
    }

    CategoricalCrossEntropyWithSoftmax loss_fn;

    top_(&w_time_par1);
    CNN_Weights_Par_Thread w_res = cnn_reduce(X, Y, loss_fn, alpha, seuil, n_epoch, n_threads);
    top_(&w_time_par2);

    work_time_par = cpu_time_(w_time_par1, w_time_par2);
    return w_res;
}
