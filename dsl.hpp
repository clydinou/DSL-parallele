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

// ─────────────────────────────────────────────
//  Interface & Structures
// ─────────────────────────────────────────────

class LossFunction {
public:
    virtual ~LossFunction() = default;
    virtual double forward(const Matrice& logits, const std::vector<int>& targets) = 0;
    virtual Matrice backward() = 0;
    // Cree une copie independante de cette fonction de perte. Indispensable
    // pour que chaque thread ait sa propre instance (avec son propre etat
    // interne logits_cached/targets_cached/probs) au lieu de partager le
    // meme objet, ce qui provoquait une data race entre threads.
    virtual LossFunction* clone() const = 0;
};

typedef struct {
    int index;
    int debut;
    int fin;
    double alpha;
    double cout;          // Loss locale moyenne accumulée par le thread
    int batch_size;        // Taille des mini-batchs traités par le thread
    const Tensor* X;             
    const std::vector<int>* Y;   
    const int* ordre;
    LossFunction* loss;   // Fonction de perte

    // Poids de depart pour cette epoque (federated averaging).
    // nullptr => initialisation aleatoire (utilise seulement a la 1ere epoque) ;
    // sinon, chaque thread reprend la moyenne des poids obtenus a l'epoque precedente.
    const std::vector<Tensor>* W_conv_init;
    const std::vector<std::vector<double>>* b_conv_init;
    const std::vector<Matrice>* W_dense_init;
    const std::vector<std::vector<double>>* b_dense_init;

    std::vector<Tensor> W_conv_local;    // Poids de toutes les couches Conv2D
    std::vector<std::vector<double>> b_conv_local; // Biais de toutes les couches Conv2D
    std::vector<Matrice> W_dense_local;   // Poids de toutes les couches Dense
    std::vector<std::vector<double>> b_dense_local; // Biais de toutes les couches Dense
} Arg_CNN;

typedef struct {
    std::vector<Arg_CNN> threads_results; 
    Matrice erreurs_threads;
} CNN_Weights_Par_Thread;

// ─────────────────────────────────────────────
//  Utilitaires
// ─────────────────────────────────────────────

int* range(int size);
int select_random_number(int min, int max);
void shuffle(int* array, int n);

// ─────────────────────────────────────────────
//  Fonctions d'Activation & Coût
// ─────────────────────────────────────────────

Matrice sigmoid(Matrice A);
Matrice softmax(Matrice A);

// ─────────────────────────────────────────────
//  Gestion des Tenseurs
// ─────────────────────────────────────────────

Tensor create_tensor(int n, int c, int h, int w, double fill = 0.0);
Tensor pad_tensor(const Tensor& x, int pad);

// ─────────────────────────────────────────────
//  Couches du Réseau (Classes CNN)
// ─────────────────────────────────────────────

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

class ReLU {
public:
    Tensor x_cached;

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

class DenseReLU {
public:
    Matrice x_cached;

    Matrice forward(const Matrice& x);
    Matrice backward(const Matrice& dout, double lr);
};

// ─────────────────────────────────────────────
//  Fonction de Perte (Loss)
// ─────────────────────────────────────────────

class CategoricalCrossEntropyWithSoftmax : public LossFunction {
public:
    Matrice logits_cached;
    std::vector<int> targets_cached;
    Matrice probs;

    double forward(const Matrice& logits, const std::vector<int>& targets) override;
    Matrice backward() override;
    LossFunction* clone() const override { return new CategoricalCrossEntropyWithSoftmax(*this); }
};

// ─────────────────────────────────────────────
//  Multi-threading & Architecture Parallèle
// ─────────────────────────────────────────────

void* cnn_local(void* a);
CNN_Weights_Par_Thread cnn_reduce(Tensor X, std::vector<int> Y, LossFunction& loss, double alpha, double seuil, int n_epoch, int n_threads, int batch_size = 32);

#endif // _DSL_HPP