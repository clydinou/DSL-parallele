#include "dsl.hpp"
#include <pthread.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <memory>

// ─────────────────────────────────────────────
//  Utilitaires
// ─────────────────────────────────────────────

int* range(int size) {
    int* r = new int[size];
    for (int i = 0; i < size; i++) r[i] = i;
    return r;
}

int select_random_number(int min, int max) {
    // std::mt19937 thread_local : par precaution, meme si cette fonction n'est
    // aujourd'hui appelee que depuis le thread principal (via shuffle() dans
    // cnn_reduce). Evite de reintroduire une dependance a rand()/RAND_MAX,
    // non thread-safe, si la fonction venait a etre appelee depuis un thread
    // de travail a l'avenir.
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(min, max);
    return dist(gen);
}

void shuffle(int* array, int n) {
    if (n > 1) {
        for (int i = 0; i < n; i++) {
            int j = select_random_number(i, n - 1);
            std::swap(array[i], array[j]);
        }
    }
}

// ─────────────────────────────────────────────
//  Activations (Matrice)
// ─────────────────────────────────────────────

Matrice sigmoid(Matrice A) {
    Matrice out = zeros(A.n, A.m);
    for (int i = 0; i < A.n; i++)
        for (int j = 0; j < A.m; j++)
            out.contenu[i][j] = 1.0 / (1.0 + exp(-(double)A.contenu[i][j]));
    return out;
}

Matrice softmax(Matrice A) {
    Matrice out = zeros(A.n, A.m);
    for (int i = 0; i < A.n; i++) {
        double s = 0.0;
        for (int j = 0; j < A.m; j++) s += exp((double)A.contenu[i][j]);
        for (int j = 0; j < A.m; j++)
            out.contenu[i][j] = exp((double)A.contenu[i][j]) / s;
    }
    return out;
}

Matrice DenseReLU::forward(const Matrice& x) {
    x_cached = x;
    Matrice out = zeros(x.n, x.m);
    for (int i = 0; i < x.n; i++)
        for (int j = 0; j < x.m; j++)
            out.contenu[i][j] = (x.contenu[i][j] > 0.0) ? x.contenu[i][j] : 0.0;
    return out;
}

Matrice DenseReLU::backward(const Matrice& dout, double lr) {
    (void)lr; // DenseReLU n'a pas de poids a mettre a jour
    Matrice dx = zeros(dout.n, dout.m);
    for (int i = 0; i < dout.n; i++)
        for (int j = 0; j < dout.m; j++)
            dx.contenu[i][j] = (x_cached.contenu[i][j] > 0.0) ? dout.contenu[i][j] : 0.0;
    return dx;
}

// ─────────────────────────────────────────────
//  Utilitaires Tensor
// ─────────────────────────────────────────────

Tensor create_tensor(int n, int c, int h, int w, double fill) {
    Tensor T = initialisation_tensor(n, c, h, w);
    if (fill != 0.0) {
        for (int i = 0; i < n; i++)
            for (int k = 0; k < c; k++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        T.data[i][k][y][x] = (double)fill;
    }
    return T;
}

Tensor pad_tensor(const Tensor& x, int pad) {
    if (pad == 0) return x;
    int n = x.n, c = x.c, h = x.h, w = x.w;
    Tensor padded = create_tensor(n, c, h + 2 * pad, w + 2 * pad, 0.0);
    for (int b = 0; b < n; b++)
        for (int ch = 0; ch < c; ch++)
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                    padded.data[b][ch][i + pad][j + pad] = x.data[b][ch][i][j];
    return padded;
}

// ─────────────────────────────────────────────
//  Couches du Réseau (Classes CNN)
// ─────────────────────────────────────────────

// ----- Conv2D -----

Conv2D::Conv2D(int in_c, int out_c, int k_size, int str, int pad)
    : in_channels(in_c), out_channels(out_c), kernel_size(k_size),
      stride(str), padding(pad)
{
    weights = initialisation_tensor(out_channels, in_channels, kernel_size, kernel_size);
    randomize_tensor(&weights, -0.1f, 0.1f);
    biases.assign(out_channels, 0.0);
    dweights = initialisation_tensor(out_channels, in_channels, kernel_size, kernel_size);
    dbiases.assign(out_channels, 0.0);
}

Tensor Conv2D::forward(const Tensor& x) {
    x_cached = x;
    Tensor xp = pad_tensor(x, padding);

    int n = xp.n;
    int out_h = (xp.h - kernel_size) / stride + 1;
    int out_w = (xp.w - kernel_size) / stride + 1;
    Tensor out = create_tensor(n, out_channels, out_h, out_w, 0.0);

    for (int b = 0; b < n; b++)
        for (int oc = 0; oc < out_channels; oc++)
            for (int oy = 0; oy < out_h; oy++)
                for (int ox = 0; ox < out_w; ox++) {
                    double s = (double)biases[oc];
                    for (int ic = 0; ic < in_channels; ic++)
                        for (int ky = 0; ky < kernel_size; ky++)
                            for (int kx = 0; kx < kernel_size; kx++) {
                                int iy = oy * stride + ky;
                                int ix = ox * stride + kx;
                                s += xp.data[b][ic][iy][ix] * weights.data[oc][ic][ky][kx];
                            }
                    out.data[b][oc][oy][ox] = s;
                }
    return out;
}

Tensor Conv2D::backward(const Tensor& dout, double lr) {
    Tensor xp = pad_tensor(x_cached, padding);
    int n = xp.n;
    int out_h = dout.h, out_w = dout.w;

    Tensor dW = initialisation_tensor(out_channels, in_channels, kernel_size, kernel_size);
    std::vector<double> dB(out_channels, 0.0);
    Tensor dXp = create_tensor(n, in_channels, xp.h, xp.w, 0.0);

    for (int b = 0; b < n; b++)
        for (int oc = 0; oc < out_channels; oc++)
            for (int oy = 0; oy < out_h; oy++)
                for (int ox = 0; ox < out_w; ox++) {
                    double grad = dout.data[b][oc][oy][ox];
                    dB[oc] += (double)grad;
                    for (int ic = 0; ic < in_channels; ic++)
                        for (int ky = 0; ky < kernel_size; ky++)
                            for (int kx = 0; kx < kernel_size; kx++) {
                                int iy = oy * stride + ky;
                                int ix = ox * stride + kx;
                                dW.data[oc][ic][ky][kx] += grad * xp.data[b][ic][iy][ix];
                                dXp.data[b][ic][iy][ix] += grad * weights.data[oc][ic][ky][kx];
                            }
                }

    dweights = dW;
    dbiases  = dB;

    // Mise à jour des poids (descente de gradient)
    for (int oc = 0; oc < out_channels; oc++) {
        for (int ic = 0; ic < in_channels; ic++)
            for (int ky = 0; ky < kernel_size; ky++)
                for (int kx = 0; kx < kernel_size; kx++)
                    weights.data[oc][ic][ky][kx] -= (double)lr * dW.data[oc][ic][ky][kx];
        biases[oc] -= lr * dB[oc];
    }

    // Retrait du padding pour retrouver la forme de l'entrée d'origine
    Tensor dX = create_tensor(n, in_channels, x_cached.h, x_cached.w, 0.0);
    for (int b = 0; b < n; b++)
        for (int ic = 0; ic < in_channels; ic++)
            for (int y = 0; y < x_cached.h; y++)
                for (int x = 0; x < x_cached.w; x++)
                    dX.data[b][ic][y][x] = dXp.data[b][ic][y + padding][x + padding];

    return dX;
}

// ----- ReLU (Tensor) -----

Tensor ReLU::forward(const Tensor& x) {
    x_cached = x;
    Tensor out = create_tensor(x.n, x.c, x.h, x.w, 0.0);
    for (int b = 0; b < x.n; b++)
        for (int ch = 0; ch < x.c; ch++)
            for (int y = 0; y < x.h; y++)
                for (int col = 0; col < x.w; col++)
                    out.data[b][ch][y][col] = (x.data[b][ch][y][col] > 0.0) ? x.data[b][ch][y][col] : 0.0;
    return out;
}

Tensor ReLU::backward(const Tensor& dout, double lr) {
    (void)lr; // pas de poids
    Tensor dx = create_tensor(dout.n, dout.c, dout.h, dout.w, 0.0);
    for (int b = 0; b < dout.n; b++)
        for (int ch = 0; ch < dout.c; ch++)
            for (int y = 0; y < dout.h; y++)
                for (int col = 0; col < dout.w; col++)
                    dx.data[b][ch][y][col] = (x_cached.data[b][ch][y][col] > 0.0) ? dout.data[b][ch][y][col] : 0.0;
    return dx;
}

// ----- MaxPool2D -----

MaxPool2D::MaxPool2D(int p_size, int str) : pool_size(p_size), stride(str) {}

Tensor MaxPool2D::forward(const Tensor& x) {
    x_cached = x;
    max_indices.clear();

    int out_h = (x.h - pool_size) / stride + 1;
    int out_w = (x.w - pool_size) / stride + 1;
    Tensor out = create_tensor(x.n, x.c, out_h, out_w, 0.0);

    for (int b = 0; b < x.n; b++)
        for (int ch = 0; ch < x.c; ch++)
            for (int oy = 0; oy < out_h; oy++)
                for (int ox = 0; ox < out_w; ox++) {
                    double best = -1e300;
                    int best_y = oy * stride, best_x = ox * stride;
                    for (int py = 0; py < pool_size; py++)
                        for (int px = 0; px < pool_size; px++) {
                            int iy = oy * stride + py;
                            int ix = ox * stride + px;
                            double v = x.data[b][ch][iy][ix];
                            if (v > best) { best = v; best_y = iy; best_x = ix; }
                        }
                    out.data[b][ch][oy][ox] = best;
                    max_indices[std::make_tuple(b, ch, oy, ox)] = std::make_pair(best_y, best_x);
                }
    return out;
}

Tensor MaxPool2D::backward(const Tensor& dout, double lr) {
    (void)lr; // pas de poids
    Tensor dx = create_tensor(x_cached.n, x_cached.c, x_cached.h, x_cached.w, 0.0);
    for (int b = 0; b < dout.n; b++)
        for (int ch = 0; ch < dout.c; ch++)
            for (int oy = 0; oy < dout.h; oy++)
                for (int ox = 0; ox < dout.w; ox++) {
                    auto it = max_indices.find(std::make_tuple(b, ch, oy, ox));
                    if (it != max_indices.end()) {
                        int iy = it->second.first, ix = it->second.second;
                        dx.data[b][ch][iy][ix] += dout.data[b][ch][oy][ox];
                    }
                }
    return dx;
}

// ----- Flatten -----

Matrice Flatten::forward(const Tensor& x) {
    orig_shape = { x.n, x.c, x.h, x.w };
    return tensor_to_matrice(&x);
}

Tensor Flatten::backward(const Matrice& dout, double lr) {
    (void)lr; // pas de poids
    return matrice_to_tensor(&dout, orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3]);
}

// ----- Dense -----

Dense::Dense(int in_f, int out_f) : in_features(in_f), out_features(out_f) {
    weights = random_matrice(in_f, out_f);
    produit_scalaire_matrice(&weights, 0.2);
    somme_scalaire_matrice(&weights, -0.1); // poids initiaux dans [-0.1, 0.1]
    biases.assign(out_f, 0.0);
}

Matrice Dense::forward(const Matrice& x) {
    x_cached = x;
    Matrice out = productmatriciel(x, weights); // (n,in) x (in,out) = (n,out)
    for (int i = 0; i < out.n; i++)
        for (int j = 0; j < out.m; j++)
            out.contenu[i][j] += (double)biases[j];
    return out;
}

Matrice Dense::backward(const Matrice& dout, double lr) {
    Matrice xt = transpose(x_cached);
    Matrice dW = productmatriciel(xt, dout); // (in,n) x (n,out) = (in,out)

    std::vector<double> dB(out_features, 0.0);
    for (int j = 0; j < dout.m; j++) {
        double s = 0.0;
        for (int i = 0; i < dout.n; i++) s += dout.contenu[i][j];
        dB[j] = (double)s;
    }

    Matrice Wt = transpose(weights);
    Matrice dx = productmatriciel(dout, Wt); // (n,out) x (out,in) = (n,in)

    // Mise à jour des poids (descente de gradient)
    for (int i = 0; i < in_features; i++)
        for (int j = 0; j < out_features; j++)
            weights.contenu[i][j] -= (double)lr * dW.contenu[i][j];
    for (int j = 0; j < out_features; j++)
        biases[j] -= lr * dB[j];

    return dx;
}

// ─────────────────────────────────────────────
//  Fonction de Perte (Loss)
// ─────────────────────────────────────────────

double CategoricalCrossEntropyWithSoftmax::forward(const Matrice& logits, const std::vector<int>& targets) {
    logits_cached  = logits;
    targets_cached = targets;
    probs = softmax(logits);

    double loss = 0.0;
    int n = logits.n;
    for (int i = 0; i < n; i++) {
        double p = probs.contenu[i][targets[i]];
        if (p < 1e-15) p = 1e-15;
        loss += -log(p);
    }
    return (double)(loss / n);
}

Matrice CategoricalCrossEntropyWithSoftmax::backward() {
    int n = probs.n, m = probs.m;
    Matrice d = zeros(n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            d.contenu[i][j] = probs.contenu[i][j];
        d.contenu[i][targets_cached[i]] -= 1.0;
        for (int j = 0; j < m; j++)
            d.contenu[i][j] /= (double)n;
    }
    return d;
}

// =====================================================================
// Fonction thread : cnn_local
// =====================================================================

void* cnn_local(void* a) {
    Arg_CNN* ar = (Arg_CNN*) a;
    int debut = ar->debut;
    int fin   = ar->fin;
    int total_local = fin - debut;

    if (total_local <= 0) {
        ar->cout = 0.0;
        return NULL;
    }

    const int* ordre = ar->ordre;
    int c = ar->X->c, h = ar->X->h, w = ar->X->w;

    // Taille des mini-batchs : bornée par la tranche du thread pour rester valide
    // meme si la tranche est plus petite que le mini-batch demande.
    int mini_batch = ar->batch_size > 0 ? ar->batch_size : total_local;
    if (mini_batch > total_local) mini_batch = total_local;

    // Architecture CNN locale : creee une seule fois, ses poids persistent
    // et sont mis a jour au fil de tous les mini-batchs de cette epoque.
    Conv2D conv1(c, 8, 3, 1, 1);    // padding=1 pour preserver la taille
    ReLU   relu1;
    MaxPool2D pool1(2, 2);

    Conv2D conv2(8, 16, 3, 1, 1);
    ReLU   relu2;
    MaxPool2D pool2(2, 2);

    Flatten flatten;

    int h2 = h / 4, w2 = w / 4;
    int flat_features = 16 * h2 * w2;
    Dense     dense1(flat_features, 64);
    DenseReLU relu_dense1;
    Dense     dense_out(64, 10);

    // Instance de la fonction de perte PROPRE A CE THREAD : ar->loss pointe
    // vers l'unique objet partage par tous les threads (cree une seule fois
    // dans CNNAlgorithm), donc l'utiliser directement provoquerait une data
    // race sur son etat interne (logits_cached/targets_cached/probs) des que
    // plusieurs threads appellent forward()/backward() en parallele.
    std::unique_ptr<LossFunction> loss_local(ar->loss->clone());

    // Reprise des poids de l'epoque precedente si disponibles (federated averaging).
    // A la 1ere epoque, ar->*_init valent nullptr : on garde l'initialisation aleatoire.
    if (ar->W_conv_init && ar->b_conv_init) {
        conv1.weights = (*ar->W_conv_init)[0];
        conv1.biases  = (*ar->b_conv_init)[0];
        conv2.weights = (*ar->W_conv_init)[1];
        conv2.biases  = (*ar->b_conv_init)[1];
    }
    if (ar->W_dense_init && ar->b_dense_init) {
        dense1.weights    = (*ar->W_dense_init)[0];
        dense1.biases     = (*ar->b_dense_init)[0];
        dense_out.weights = (*ar->W_dense_init)[1];
        dense_out.biases  = (*ar->b_dense_init)[1];
    }

    double cout_total = 0.0;
    int nb_mini_batches = 0;

    // On ne materialise jamais toute la tranche du thread d'un coup :
    // seul un mini-batch a la fois est extrait de ar->X, ce qui borne le
    // pic memoire a mini_batch images au lieu de total_local.
    for (int mb_start = debut; mb_start < fin; mb_start += mini_batch) {
        int mb_end  = std::min(mb_start + mini_batch, fin);
        int mb_size = mb_end - mb_start;

        Tensor X_batch = create_tensor(mb_size, c, h, w, 0.0);
        std::vector<int> Y_batch(mb_size);

        for (int i = mb_start; i < mb_end; i++) {
            int idx = ordre[i];
            int local_idx = i - mb_start;
            Y_batch[local_idx] = (*ar->Y)[idx];
            for (int ch = 0; ch < c; ch++)
                for (int r = 0; r < h; r++)
                    for (int col = 0; col < w; col++)
                        X_batch.data[local_idx][ch][r][col] = ar->X->data[idx][ch][r][col];
        }

        // ----- FORWARD -----
        Tensor   out_conv1   = conv1.forward(X_batch);
        Tensor   out_relu1   = relu1.forward(out_conv1);
        Tensor   out_pool1   = pool1.forward(out_relu1);

        Tensor   out_conv2   = conv2.forward(out_pool1);
        Tensor   out_relu2   = relu2.forward(out_conv2);
        Tensor   out_pool2   = pool2.forward(out_relu2);

        Matrice  out_flat    = flatten.forward(out_pool2);
        Matrice  out_dense1  = dense1.forward(out_flat);
        Matrice  out_rdense1 = relu_dense1.forward(out_dense1);
        Matrice  logits      = dense_out.forward(out_rdense1);

        double batch_loss = loss_local->forward(logits, Y_batch);
        cout_total += batch_loss;
        nb_mini_batches++;

        // ----- BACKWARD -----
        Matrice dlogits     = loss_local->backward();
        Matrice ddense_out  = dense_out.backward(dlogits,     ar->alpha);
        Matrice drelu_d1    = relu_dense1.backward(ddense_out, ar->alpha);
        Matrice ddense1     = dense1.backward(drelu_d1,       ar->alpha);
        Tensor  dflat       = flatten.backward(ddense1,       ar->alpha);
        Tensor  dpool2      = pool2.backward(dflat,           ar->alpha);
        Tensor  drelu2      = relu2.backward(dpool2,          ar->alpha);
        Tensor  dconv2      = conv2.backward(drelu2,          ar->alpha);
        Tensor  dpool1      = pool1.backward(dconv2,          ar->alpha);
        Tensor  drelu1      = relu1.backward(dpool1,          ar->alpha);
        /*Tensor dconv1 =*/   conv1.backward(drelu1,          ar->alpha);
    }

    // Loss moyenne sur tous les mini-batchs traites par ce thread
    ar->cout = (nb_mini_batches > 0) ? (cout_total / nb_mini_batches) : 0.0;

    // Sauvegarde des poids locaux (etat final apres tous les mini-batchs)
    ar->W_conv_local.clear();
    ar->b_conv_local.clear();
    ar->W_dense_local.clear();
    ar->b_dense_local.clear();

    ar->W_conv_local.push_back(conv1.weights);
    ar->b_conv_local.push_back(conv1.biases);
    ar->W_conv_local.push_back(conv2.weights);
    ar->b_conv_local.push_back(conv2.biases);
    ar->W_dense_local.push_back(dense1.weights);
    ar->b_dense_local.push_back(dense1.biases);
    ar->W_dense_local.push_back(dense_out.weights);
    ar->b_dense_local.push_back(dense_out.biases);

    return NULL;
}

// =====================================================================
// Moyenne des poids entre threads (federated averaging)
// =====================================================================

static Tensor moyenne_tensors(const std::vector<Tensor>& vs) {
    const Tensor& ref = vs[0];
    Tensor out = create_tensor(ref.n, ref.c, ref.h, ref.w, 0.0);
    for (const Tensor& t : vs)
        for (int i = 0; i < out.n; i++)
            for (int c = 0; c < out.c; c++)
                for (int y = 0; y < out.h; y++)
                    for (int x = 0; x < out.w; x++)
                        out.data[i][c][y][x] += t.data[i][c][y][x];
    double inv = 1.0 / (double)vs.size();
    for (int i = 0; i < out.n; i++)
        for (int c = 0; c < out.c; c++)
            for (int y = 0; y < out.h; y++)
                for (int x = 0; x < out.w; x++)
                    out.data[i][c][y][x] *= inv;
    return out;
}

static Matrice moyenne_matrices(const std::vector<Matrice>& vs) {
    Matrice out = zeros(vs[0].n, vs[0].m);
    for (const Matrice& m : vs)
        for (int i = 0; i < out.n; i++)
            for (int j = 0; j < out.m; j++)
                out.contenu[i][j] += m.contenu[i][j];
    double inv = 1.0 / (double)vs.size();
    for (int i = 0; i < out.n; i++)
        for (int j = 0; j < out.m; j++)
            out.contenu[i][j] *= inv;
    return out;
}

static std::vector<double> moyenne_vecteurs(const std::vector<std::vector<double>>& vs) {
    std::vector<double> out(vs[0].size(), 0.0);
    for (const auto& v : vs)
        for (size_t i = 0; i < out.size(); i++)
            out[i] += v[i];
    double inv = 1.0 / (double)vs.size();
    for (double& x : out) x *= inv;
    return out;
}

// =====================================================================
// cnn_reduce : MapReduce multi-thread
// =====================================================================

CNN_Weights_Par_Thread cnn_reduce(Tensor X, std::vector<int> Y, LossFunction& loss,
                                   double alpha, double seuil, int n_epoch, int n_threads, int batch_size) {
    int iter = 1;
    double error = seuil + 1.0; // forcer au moins 1 époque
    int total_samples = X.n;
    int* ordre = range(total_samples);

    CNN_Weights_Par_Thread sortie_finale;
    sortie_finale.threads_results.resize(n_threads);
    sortie_finale.erreurs_threads = zeros(n_threads, 1);

    // Poids globaux partages entre les epoques (federated averaging).
    // Vides a la 1ere epoque (chaque thread part d'une init aleatoire) ;
    // a partir de l'epoque 2, tous les threads repartent de la moyenne
    // des poids obtenus par l'ensemble des threads a l'epoque precedente.
    std::vector<Tensor> global_W_conv;
    std::vector<std::vector<double>> global_b_conv;
    std::vector<Matrice> global_W_dense;
    std::vector<std::vector<double>> global_b_dense;
    bool has_global_weights = false;

    while (n_epoch >= iter && error >= seuil) {
        std::vector<pthread_t> threads(n_threads);
        std::vector<Arg_CNN>   argument(n_threads);

        shuffle(ordre, total_samples);
        int fraction = total_samples / n_threads;

        // 1. MAPPER : lancement des threads
        for (int i = 0; i < n_threads; i++) {
            Arg_CNN* ar = &argument[i];
            ar->index = i;
            ar->alpha = alpha;
            ar->batch_size = batch_size;
            ar->X     = &X;
            ar->Y     = &Y;
            ar->ordre = ordre;
            ar->loss  = &loss;
            ar->debut = i * fraction;
            ar->fin   = (i == n_threads - 1) ? total_samples : ar->debut + fraction;

            if (has_global_weights) {
                ar->W_conv_init  = &global_W_conv;
                ar->b_conv_init  = &global_b_conv;
                ar->W_dense_init = &global_W_dense;
                ar->b_dense_init = &global_b_dense;
            } else {
                ar->W_conv_init  = nullptr;
                ar->b_conv_init  = nullptr;
                ar->W_dense_init = nullptr;
                ar->b_dense_init = nullptr;
            }

            pthread_create(&threads[i], NULL, cnn_local, (void*)ar);
        }

        if (iter == 1)
            std::cout << "Lancement de " << n_threads << " threads..." << std::endl;

        // Attente
        for (int i = 0; i < n_threads; i++)
            pthread_join(threads[i], NULL);

        // 2. REDUCER
        double cout_total = 0.0;
        for (int k = 0; k < n_threads; k++) {
            cout_total += argument[k].cout;
            sortie_finale.threads_results[k] = argument[k];
            sortie_finale.erreurs_threads.contenu[k][0] = (double)argument[k].cout;
        }
        error = cout_total / n_threads; // loss moyenne par thread

        // 2bis. Fusion des poids (federated averaging) : la moyenne des poids
        // obtenus par tous les threads devient le point de depart commun
        // de l'epoque suivante, au lieu de repartir d'une init aleatoire.
        std::vector<Tensor> new_W_conv(2);
        std::vector<std::vector<double>> new_b_conv(2);
        for (int layer = 0; layer < 2; layer++) {
            std::vector<Tensor> Ws;
            std::vector<std::vector<double>> Bs;
            for (int k = 0; k < n_threads; k++) {
                Ws.push_back(argument[k].W_conv_local[layer]);
                Bs.push_back(argument[k].b_conv_local[layer]);
            }
            new_W_conv[layer] = moyenne_tensors(Ws);
            new_b_conv[layer] = moyenne_vecteurs(Bs);
        }

        std::vector<Matrice> new_W_dense(2);
        std::vector<std::vector<double>> new_b_dense(2);
        for (int layer = 0; layer < 2; layer++) {
            std::vector<Matrice> Ws;
            std::vector<std::vector<double>> Bs;
            for (int k = 0; k < n_threads; k++) {
                Ws.push_back(argument[k].W_dense_local[layer]);
                Bs.push_back(argument[k].b_dense_local[layer]);
            }
            new_W_dense[layer] = moyenne_matrices(Ws);
            new_b_dense[layer] = moyenne_vecteurs(Bs);
        }

        global_W_conv  = std::move(new_W_conv);
        global_b_conv  = std::move(new_b_conv);
        global_W_dense = std::move(new_W_dense);
        global_b_dense = std::move(new_b_dense);
        has_global_weights = true;

        // 3. SUIVI
        if (iter % 10 == 0 || iter == 1)
            std::cout << "Epoque " << iter << " / " << n_epoch << " - Loss: " << error << std::endl;

        if (iter == n_epoch || error < seuil) {
            std::cout << "\n--- Loss de chaque thread a la derniere epoque ---" << std::endl;
            for (int k = 0; k < n_threads; k++)
                std::cout << "Thread [" << k << "] : " << argument[k].cout << std::endl;
        }

        iter++;
    }

    delete[] ordre;
    return sortie_finale;
}