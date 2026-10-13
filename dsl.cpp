#include "dsl.hpp"
#include <pthread.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <memory>

int* range(int size) {
    int* r = new int[size];
    for (int i = 0; i < size; i++) r[i] = i;
    return r;
}

int select_random_number(int min, int max) {
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

Matrice sigmoid(Matrice A) {
    Matrice out = zeros(A.n, A.m);
    for (int i = 0; i < A.n; i++)
        for (int j = 0; j < A.m; j++)
            out.contenu[i][j] = sigmoid(A.contenu[i][j]);
    return out;
}

Matrice softmax(Matrice A) {
    Matrice out = zeros(A.n, A.m);
    for (int i = 0; i < A.n; i++) {
        double max_val = -1e300;
        for (int j = 0; j < A.m; j++)
            if (A.contenu[i][j] > max_val) max_val = A.contenu[i][j];

        double s = 0.0;
        for (int j = 0; j < A.m; j++) s += std::exp(A.contenu[i][j] - max_val);
        for (int j = 0; j < A.m; j++)
            out.contenu[i][j] = std::exp(A.contenu[i][j] - max_val) / s;
    }
    return out;
}

Matrice ReLU::forward(const Matrice& x) {
    x_cached = x;
    Matrice out = zeros(x.n, x.m);
    for (int i = 0; i < x.n; i++)
        for (int j = 0; j < x.m; j++)
            out.contenu[i][j] = (x.contenu[i][j] > 0.0) ? x.contenu[i][j] : 0.0;
    return out;
}

Matrice ReLU::backward(const Matrice& dout, double lr) {
    (void)lr;
    Matrice dx = zeros(dout.n, dout.m);
    for (int i = 0; i < dout.n; i++)
        for (int j = 0; j < dout.m; j++)
            dx.contenu[i][j] = (x_cached.contenu[i][j] > 0.0) ? dout.contenu[i][j] : 0.0;
    return dx;
}

Tensor create_tensor(int n, int c, int h, int w, double fill) {
    Tensor T = initialisation_tensor(n, c, h, w);
    if (fill != 0.0) {
        for (int i = 0; i < n; i++)
            for (int k = 0; k < c; k++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        T.data[i][k][y][x] = fill;
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
                    double s = biases[oc];
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
                    dB[oc] += grad;
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

    for (int oc = 0; oc < out_channels; oc++) {
        for (int ic = 0; ic < in_channels; ic++)
            for (int ky = 0; ky < kernel_size; ky++)
                for (int kx = 0; kx < kernel_size; kx++)
                    weights.data[oc][ic][ky][kx] -= lr * dW.data[oc][ic][ky][kx];
        biases[oc] -= lr * dB[oc];
    }

    Tensor dX = create_tensor(n, in_channels, x_cached.h, x_cached.w, 0.0);
    for (int b = 0; b < n; b++)
        for (int ic = 0; ic < in_channels; ic++)
            for (int y = 0; y < x_cached.h; y++)
                for (int x = 0; x < x_cached.w; x++)
                    dX.data[b][ic][y][x] = dXp.data[b][ic][y + padding][x + padding];

    return dX;
}

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
    (void)lr;
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

Matrice Flatten::forward(const Tensor& x) {
    orig_shape = { x.n, x.c, x.h, x.w };
    return tensor_to_matrice(&x);
}

Tensor Flatten::backward(const Matrice& dout, double lr) {
    (void)lr;
    return matrice_to_tensor(&dout, orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3]);
}

Dense::Dense(int in_f, int out_f) : in_features(in_f), out_features(out_f) {
    weights = random_matrice(in_f, out_f);
    produit_scalaire_matrice(&weights, 0.2);
    somme_scalaire_matrice(&weights, -0.1);
    biases.assign(out_f, 0.0);
}

Matrice Dense::forward(const Matrice& x) {
    x_cached = x;
    Matrice out = productmatriciel(x, weights);
    for (int i = 0; i < out.n; i++)
        for (int j = 0; j < out.m; j++)
            out.contenu[i][j] += biases[j];
    return out;
}

Matrice Dense::backward(const Matrice& dout, double lr) {
    Matrice xt = transpose(x_cached);
    Matrice dW = productmatriciel(xt, dout);
    std::vector<double> dB(out_features, 0.0);
    for (int j = 0; j < dout.m; j++) {
        double s = 0.0;
        for (int i = 0; i < dout.n; i++) s += dout.contenu[i][j];
        dB[j] = s;
    }
    Matrice Wt = transpose(weights);
    Matrice dx = productmatriciel(dout, Wt);
    for (int i = 0; i < in_features; i++)
        for (int j = 0; j < out_features; j++)
            weights.contenu[i][j] -= lr * dW.contenu[i][j];
    for (int j = 0; j < out_features; j++)
        biases[j] -= lr * dB[j];
    return dx;
}

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
    return loss / n;
}

Matrice CategoricalCrossEntropyWithSoftmax::backward() {
    int n = probs.n, m = probs.m;
    Matrice d = zeros(n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) d.contenu[i][j] = probs.contenu[i][j];
        d.contenu[i][targets_cached[i]] -= 1.0;
        for (int j = 0; j < m; j++) d.contenu[i][j] /= (double)n;
    }
    return d;
}

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
    int mini_batch = ar->batch_size > 0 ? ar->batch_size : total_local;
    if (mini_batch > total_local) mini_batch = total_local;
    Conv2D conv1(c, 8, 3, 1, 1);    
    MaxPool2D pool1(2, 2);
    Conv2D conv2(8, 16, 3, 1, 1);
    MaxPool2D pool2(2, 2);
    Flatten flatten;
    int h2 = h / 4, w2 = w / 4;
    int flat_features = 16 * h2 * w2;
    Dense     dense1(flat_features, 64);
    ReLU   relu;
    Dense     dense_out(64, 10);
    std::unique_ptr<LossFunction> loss_local(ar->loss->clone());
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
        Tensor   out_conv1 = conv1.forward(X_batch);
        Tensor   out_pool1 = pool1.forward(out_conv1);
        Tensor   out_conv2 = conv2.forward(out_pool1);
        Tensor   out_pool2 = pool2.forward(out_conv2);
        Matrice  out_flat = flatten.forward(out_pool2);
        Matrice  out_dense1 = dense1.forward(out_flat);
        Matrice  out_relu = relu.forward(out_dense1);
        Matrice  logits = dense_out.forward(out_relu);
        double batch_loss = loss_local->forward(logits, Y_batch);
        cout_total += batch_loss;
        nb_mini_batches++;
        Matrice dlogits = loss_local->backward();
        Matrice ddense_out = dense_out.backward(dlogits, ar->alpha);
        Matrice drelu = relu.backward(ddense_out, ar->alpha);
        Matrice ddense1 = dense1.backward(drelu, ar->alpha);
        Tensor  dflat = flatten.backward(ddense1, ar->alpha);
        Tensor  dpool2 = pool2.backward(dflat, ar->alpha);
        Tensor  dconv2 = conv2.backward(dpool2, ar->alpha);
        Tensor  dpool1  = pool1.backward(dconv2, ar->alpha);
        conv1.backward(dpool1, ar->alpha);
    }
    ar->cout = (nb_mini_batches > 0) ? (cout_total / nb_mini_batches) : 0.0;
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

Par_CNN_thread cnn_reduce(Tensor X, std::vector<int> Y, LossFunction& loss,
                          double alpha, double seuil, int n_epoch, int n_threads, int batch_size) {
    int iter = 1;
    double error = seuil + 1.0; 
    int total_samples = X.n;
    int* ordre = range(total_samples);
    Par_CNN_thread sortie_finale;
    sortie_finale.threads_results.resize(n_threads);
    sortie_finale.erreurs_threads = zeros(n_threads, 1);
    std::vector<Tensor> global_W_conv;
    std::vector<std::vector<double>> global_b_conv;
    std::vector<Matrice> global_W_dense;
    std::vector<std::vector<double>> global_b_dense;
    bool has_global_weights = false;

    while (n_epoch >= iter && error >= seuil) {
        std::vector<pthread_t> threads(n_threads);
        std::vector<Arg_CNN> argument(n_threads);
        shuffle(ordre, total_samples);
        int fraction = total_samples / n_threads;

        for (int i = 0; i < n_threads; i++) {
            Arg_CNN* ar = &argument[i];
            ar->index = i;
            ar->alpha = alpha;
            ar->batch_size = batch_size;
            ar->X = &X;
            ar->Y = &Y;
            ar->ordre = ordre;
            ar->loss = &loss;
            ar->debut = i * fraction;
            ar->fin = (i == n_threads - 1) ? total_samples : ar->debut + fraction;
            if (has_global_weights) {
                ar->W_conv_init = &global_W_conv;
                ar->b_conv_init = &global_b_conv;
                ar->W_dense_init = &global_W_dense;
                ar->b_dense_init = &global_b_dense;
            } else {
                ar->W_conv_init = nullptr;
                ar->b_conv_init = nullptr;
                ar->W_dense_init = nullptr;
                ar->b_dense_init = nullptr;
            }
            pthread_create(&threads[i], NULL, cnn_local, (void*)ar);
        }

        if (iter == 1)
            std::cout << "Lancement de " << n_threads << " threads..." << std::endl;

        for (int i = 0; i < n_threads; i++)
            pthread_join(threads[i], NULL);

        double cout_total = 0.0;
        for (int k = 0; k < n_threads; k++) {
            cout_total += argument[k].cout;
            sortie_finale.threads_results[k] = argument[k];
            sortie_finale.erreurs_threads.contenu[k][0] = argument[k].cout;
        }
        error = cout_total / n_threads; 

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

        if (iter % 10 == 0 || iter == 1)
            std::cout << "Epoque " << iter << " / " << n_epoch << " - Loss: " << error << std::endl;

        if (iter == n_epoch || error < seuil) {
            std::cout << "\n--- Loss de chaque thread a la derniere epoque ---" << std::endl;
            for (int k = 0; k < n_threads; k++)
                std::cout << "Thread [" << k << "] : " << argument[k].cout << std::endl;
        }

        iter++;
    }

    sortie_finale.final_W_conv  = global_W_conv;
    sortie_finale.final_b_conv  = global_b_conv;
    sortie_finale.final_W_dense = global_W_dense;
    sortie_finale.final_b_dense = global_b_dense;

    delete[] ordre;
    return sortie_finale;
}

// =====================================================================
//                             LSTM
// =====================================================================

void* lstm_local(void* a) {
    Arg_LSTM* args = (Arg_LSTM*) a;

    int hs = args->hidden_size;
    int input_dim = args->input_size;
    int T = args->seq_len;
    int target_col = args->target_col;
    args->dW_ih_local = zeros(4 * hs, input_dim + hs);
    args->db_ih_local = zeros(4 * hs, 1);
    args->dW_hy_local = zeros(1, hs);
    args->db_y_local  = zeros(1, 1);
    args->cout = 0.0;

    struct StepCache {
        Matrice f_t, i_t, g_t, o_t;
        Matrice c_t, h_t;
        Matrice c_prev;
        Matrice x_concat;
        double target_y;
        double y_pred;
    };
    std::vector<StepCache> cache(T);

    int samples_count = 0;

    for (int i = args->debut; i < args->fin; i++) {
        int idx_seq = args->ordre[i];
        samples_count++;

        Matrice h_t = zeros(hs, 1);
        Matrice c_t = zeros(hs, 1);

        // 1. FORWARD PASS
        for (int t = 0; t < T; t++) {
            Matrice c_prev = c_t;

            Matrice x_t = zeros(input_dim, 1);
            for (int k = 0; k < input_dim; k++) {
                x_t.contenu[k][0] = args->X_seq->data[idx_seq + t][k];
            }

            double target = args->X_seq->data[idx_seq + t + 1][target_col];

            Matrice x_concat;
            concatLigne(&x_concat, x_t);
            concatLigne(&x_concat, h_t);

            Matrice a_ih = productmatriciel(*args->W_ih_init, x_concat);
            
            for (int r = 0; r < 4 * hs; r++) {
                a_ih.contenu[r][0] += args->b_ih_init->contenu[r][0];
            }

            Matrice f_t = zeros(hs, 1);
            Matrice i_t = zeros(hs, 1);
            Matrice g_t = zeros(hs, 1);
            Matrice o_t = zeros(hs, 1);

            for (int k = 0; k < hs; k++) {
                f_t.contenu[k][0] = sigmoid(a_ih.contenu[k][0]);
                i_t.contenu[k][0] = sigmoid(a_ih.contenu[hs + k][0]);
                g_t.contenu[k][0] = std::tanh(a_ih.contenu[2 * hs + k][0]);
                o_t.contenu[k][0] = sigmoid(a_ih.contenu[3 * hs + k][0]);
            }

            c_t = zeros(hs, 1);
            h_t = zeros(hs, 1);
            for (int k = 0; k < hs; k++) {
                c_t.contenu[k][0] = f_t.contenu[k][0] * c_prev.contenu[k][0] + i_t.contenu[k][0] * g_t.contenu[k][0];
                h_t.contenu[k][0] = o_t.contenu[k][0] * std::tanh(c_t.contenu[k][0]);
            }

            double y_pred = args->b_y_init->contenu[0][0];
            for (int k = 0; k < hs; k++) {
                y_pred += args->W_hy_init->contenu[0][k] * h_t.contenu[k][0];
            }

            double err = y_pred - target;
            args->cout += 0.5 * (err * err);

            cache[t] = {f_t, i_t, g_t, o_t, c_t, h_t, c_prev, x_concat, target, y_pred};
        }

        // 2. BACKWARD PASS (BPTT)
        Matrice dh_next = zeros(hs, 1);
        Matrice dc_next = zeros(hs, 1);

        for (int t = T - 1; t >= 0; t--) {
            const StepCache& step = cache[t];
            double err = step.y_pred - step.target_y;

            for (int k = 0; k < hs; k++) {
                args->dW_hy_local.contenu[0][k] += err * step.h_t.contenu[k][0];
            }
            args->db_y_local.contenu[0][0] += err;

            Matrice dh_ext = zeros(hs, 1);
            for (int k = 0; k < hs; k++) {
                dh_ext.contenu[k][0] = args->W_hy_init->contenu[0][k] * err;
            }

            Matrice dh_total = zeros(hs, 1);
            for (int k = 0; k < hs; k++) {
                dh_total.contenu[k][0] = dh_ext.contenu[k][0] + dh_next.contenu[k][0];
            }

            Matrice da_f_t = zeros(hs, 1);
            Matrice da_i_t = zeros(hs, 1);
            Matrice da_g_t = zeros(hs, 1);
            Matrice da_o_t = zeros(hs, 1);
            Matrice dc_t   = zeros(hs, 1);

            for (int k = 0; k < hs; k++) {
                double tanh_ct = std::tanh(step.c_t.contenu[k][0]);
                dc_t.contenu[k][0] = dc_next.contenu[k][0] + (dh_total.contenu[k][0] * step.o_t.contenu[k][0] * (1.0 - tanh_ct * tanh_ct));

                double do_t = dh_total.contenu[k][0] * tanh_ct;
                double df_t = dc_t.contenu[k][0] * step.c_prev.contenu[k][0];
                double di_t = dc_t.contenu[k][0] * step.g_t.contenu[k][0];
                double dg_t = dc_t.contenu[k][0] * step.i_t.contenu[k][0];

                da_f_t.contenu[k][0] = df_t * sigmoid_derivative(step.f_t.contenu[k][0]);
                da_i_t.contenu[k][0] = di_t * sigmoid_derivative(step.i_t.contenu[k][0]);
                da_g_t.contenu[k][0] = dg_t * tanh_derivative(step.g_t.contenu[k][0]);
                da_o_t.contenu[k][0] = do_t * sigmoid_derivative(step.o_t.contenu[k][0]);
            }

            Matrice da_ih;
            concatLigne(&da_ih, da_f_t);
            concatLigne(&da_ih, da_i_t);
            concatLigne(&da_ih, da_g_t);
            concatLigne(&da_ih, da_o_t);

            Matrice x_concat_T = transpose(step.x_concat);
            Matrice dW_step = productmatriciel(da_ih, x_concat_T);

            for (int r = 0; r < args->dW_ih_local.n; r++) {
                for (int c = 0; c < args->dW_ih_local.m; c++) {
                    args->dW_ih_local.contenu[r][c] += dW_step.contenu[r][c];
                }
                args->db_ih_local.contenu[r][0] += da_ih.contenu[r][0];
            }

            Matrice W_ih_T = transpose(*args->W_ih_init);
            Matrice dx_concat = productmatriciel(W_ih_T, da_ih);

            for (int k = 0; k < hs; k++) {
                dh_next.contenu[k][0] = dx_concat.contenu[input_dim + k][0];
                dc_next.contenu[k][0] = dc_t.contenu[k][0] * step.f_t.contenu[k][0];
            }
        }
    }

    return nullptr;
}

double lstm_reduce(int n_threads, int input_size, int hidden_size, int seq_len, 
                   Matrice* W_ih_global, Matrice* b_ih_global, Matrice* W_hy_global, Matrice* b_y_global,
                   const Sequency& X_seq, int target_col, const std::vector<int>& ordre, double alpha) {
    int total_samples = (int)ordre.size();
    if (total_samples == 0) return 0.0;
    if (n_threads <= 0) n_threads = 1;

    int samples_per_thread = total_samples / n_threads;
    std::vector<pthread_t> threads(n_threads);
    std::vector<Arg_LSTM> args(n_threads);

    for (int t = 0; t < n_threads; ++t) {
        args[t].index = t;
        args[t].debut = t * samples_per_thread;
        args[t].fin = (t == n_threads - 1) ? total_samples : (t + 1) * samples_per_thread;
        args[t].batch_size = total_samples;
        args[t].alpha = alpha;
        args[t].input_size = input_size;
        args[t].hidden_size = hidden_size;
        args[t].seq_len = seq_len;
        args[t].target_col = target_col;
        args[t].ordre = ordre.data();
        args[t].X_seq = &X_seq;

        args[t].W_ih_init = W_ih_global;
        args[t].b_ih_init = b_ih_global;
        args[t].W_hy_init = W_hy_global;
        args[t].b_y_init  = b_y_global;

        pthread_create(&threads[t], nullptr, lstm_local, &args[t]);
    }

    double cout_total = 0.0;
    Matrice dW_total  = zeros(W_ih_global->n, W_ih_global->m);
    Matrice db_total  = zeros(b_ih_global->n, b_ih_global->m);
    Matrice dWhy_total = zeros(W_hy_global->n, W_hy_global->m);
    Matrice dby_total  = zeros(b_y_global->n, b_y_global->m);

    for (int t = 0; t < n_threads; ++t) {
        pthread_join(threads[t], nullptr);
        cout_total += args[t].cout;

        for (int r = 0; r < dW_total.n; ++r) {
            for (int c = 0; c < dW_total.m; ++c) {
                dW_total.contenu[r][c] += args[t].dW_ih_local.contenu[r][c];
            }
            db_total.contenu[r][0] += args[t].db_ih_local.contenu[r][0];
        }

        for (int c = 0; c < dWhy_total.m; ++c) {
            dWhy_total.contenu[0][c] += args[t].dW_hy_local.contenu[0][c];
        }
        dby_total.contenu[0][0] += args[t].db_y_local.contenu[0][0];
    }

    double sum_sq = 0.0;
    for (int r = 0; r < dW_total.n; ++r) {
        for (int c = 0; c < dW_total.m; ++c) {
            double v = dW_total.contenu[r][c];
            sum_sq += v * v;
        }
        double vb = db_total.contenu[r][0];
        sum_sq += vb * vb;
    }
    for (int c = 0; c < dWhy_total.m; ++c) {
        double v = dWhy_total.contenu[0][c];
        sum_sq += v * v;
    }
    sum_sq += dby_total.contenu[0][0] * dby_total.contenu[0][0];

    double norm = std::sqrt(sum_sq);
    double clip_val = 5.0;
    double scale = 1.0;
    if (norm > clip_val) {
        scale = clip_val / norm;
    }

    double factor = (alpha / total_samples) * scale;
    produit_scalaire_matrice(&dW_total, factor);
    produit_scalaire_matrice(&db_total, factor);
    produit_scalaire_matrice(&dWhy_total, factor);
    produit_scalaire_matrice(&dby_total, factor);

    *W_ih_global = difference(*W_ih_global, dW_total);
    *b_ih_global = difference(*b_ih_global, db_total);
    *W_hy_global = difference(*W_hy_global, dWhy_total);
    *b_y_global  = difference(*b_y_global, dby_total);

    // MSE normalisee par pas temporel et par echantillon
    return cout_total / (total_samples * seq_len);
}
