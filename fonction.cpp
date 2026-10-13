#include "fonction.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <iomanip>

extern TimePoint w_time_par1, w_time_par2;
extern double work_time_par;

Par_CNN_thread CNNAlgorithm(double alpha, double seuil, int n_epoch, std::string filename, int n_threads) {
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
    Par_CNN_thread w_res = cnn_reduce(X, Y, loss_fn, alpha, seuil, n_epoch, n_threads);
    top_(&w_time_par2);

    work_time_par = cpu_time_(w_time_par1, w_time_par2);
    return w_res;
}

Par_LSTM_thread LSTMAlgorithm(double alpha, double seuil, int n_epoch, std::string filename, 
                              int n_threads, int input_size, int hidden_size, int seq_len) {
    Sequency seq_data = readSequence(filename, seq_len);

    int total_timesteps = seq_data.total_samples;
    int num_features = seq_data.num_features;
    int target_col = 3; // Close

    Matrice raw_mat = zeros(total_timesteps, num_features);
    for (int i = 0; i < total_timesteps; ++i) {
        for (int j = 0; j < num_features; ++j) {
            raw_mat.contenu[i][j] = seq_data.data[i][j];
        }
    }
    raw_mat = normaliser_centrer(raw_mat);
    for (int i = 0; i < total_timesteps; ++i) {
        for (int j = 0; j < num_features; ++j) {
            seq_data.data[i][j] = raw_mat.contenu[i][j];
        }
    }

    int max_sequences = total_timesteps - seq_len - 1;
    if (max_sequences <= 0) {
        throw std::runtime_error("Jeu de donnees insuffisant pour seq_len");
    }
    int train_size = static_cast<int>(max_sequences * 0.80);

    std::vector<int> train_indices(train_size);
    std::iota(train_indices.begin(), train_indices.end(), 0);

    Matrice W_ih_global = zeros(4 * hidden_size, input_size + hidden_size);
    Matrice b_ih_global = zeros(4 * hidden_size, 1);

    double limit = std::sqrt(6.0 / (input_size + 2 * hidden_size));
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-limit, limit);

    for (int r = 0; r < W_ih_global.n; ++r) {
        for (int c = 0; c < W_ih_global.m; ++c) {
            W_ih_global.contenu[r][c] = dist(gen);
        }
    }
    for (int r = 0; r < hidden_size; ++r) {
        b_ih_global.contenu[r][0] = 1.0;
    }

    Matrice W_hy_global = zeros(1, hidden_size);
    Matrice b_y_global  = zeros(1, 1);
    double limit_y = std::sqrt(6.0 / (hidden_size + 1));
    std::uniform_real_distribution<double> dist_y(-limit_y, limit_y);
    for (int c = 0; c < hidden_size; ++c) {
        W_hy_global.contenu[0][c] = dist_y(gen);
    }

    Par_LSTM_thread res;
    res.cout_final = 0.0;
    res.epoques_effectuees = 0;

    int mini_batch_size = 32;

    top_(&w_time_par1);

    for (int epoch = 0; epoch < n_epoch; ++epoch) {
        std::shuffle(train_indices.begin(), train_indices.end(), gen);

        double epoch_loss = 0.0;
        int nb_batches = 0;

        for (int b_start = 0; b_start < train_size; b_start += mini_batch_size) {
            int b_end = std::min(b_start + mini_batch_size, train_size);
            std::vector<int> batch_order(train_indices.begin() + b_start, train_indices.begin() + b_end);

            double batch_loss = lstm_reduce(
                n_threads, input_size, hidden_size, seq_len,
                &W_ih_global, &b_ih_global, &W_hy_global, &b_y_global,
                seq_data, target_col, batch_order, alpha
            );

            epoch_loss += batch_loss;
            nb_batches++;
        }

        double loss_moyenne = epoch_loss / nb_batches;
        res.cout_final = loss_moyenne;
        res.epoques_effectuees = epoch + 1;

        if ((epoch + 1) % 5 == 0 || epoch == 0) {
            std::cout << "Epoque " << (epoch + 1) << " / " << n_epoch 
                      << " - MSE Train : " << loss_moyenne << std::endl;
        }

        if (loss_moyenne <= seuil) break;
    }

    top_(&w_time_par2);
    res.temps_execution = cpu_time_(w_time_par1, w_time_par2);

    res.W_ih = W_ih_global;
    res.b_ih = b_ih_global;
    res.W_hy = W_hy_global;
    res.b_y  = b_y_global;

    return res;
}

// ─────────────────────────────────────────────────────────────
// Evaluation du CNN sur un jeu de test
// ─────────────────────────────────────────────────────────────
void tester_modele_cnn(const Par_CNN_thread& resultat_cnn, const Tensor& X_test, const std::vector<int>& Y_test) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "        EVALUATION DU MODELE CNN        " << std::endl;
    std::cout << "========================================" << std::endl;

    int total_test = X_test.n;
    if (total_test <= 0) {
        std::cerr << "Erreur : jeu de test vide." << std::endl;
        return;
    }

    // Architecture conforme à dsl.cpp : Conv1 -> Pool1 -> Conv2 -> Pool2 -> Flatten -> Dense1 -> ReLU -> DenseOut
    Conv2D conv1(X_test.c, 8, 3, 1, 1);
    MaxPool2D pool1(2, 2);
    Conv2D conv2(8, 16, 3, 1, 1);
    MaxPool2D pool2(2, 2);
    Flatten flatten;
    int h2 = X_test.h / 4, w2 = X_test.w / 4;
    int flat_features = 16 * h2 * w2;
    Dense dense1(flat_features, 64);
    ReLU relu;
    Dense dense_out(64, 10);

    // Chargement des poids moyennés finaux
    conv1.weights     = resultat_cnn.final_W_conv[0];
    conv1.biases      = resultat_cnn.final_b_conv[0];
    conv2.weights     = resultat_cnn.final_W_conv[1];
    conv2.biases      = resultat_cnn.final_b_conv[1];
    dense1.weights    = resultat_cnn.final_W_dense[0];
    dense1.biases     = resultat_cnn.final_b_dense[0];
    dense_out.weights = resultat_cnn.final_W_dense[1];
    dense_out.biases  = resultat_cnn.final_b_dense[1];

    Tensor out_conv1 = conv1.forward(X_test);
    Tensor out_pool1 = pool1.forward(out_conv1);
    Tensor out_conv2 = conv2.forward(out_pool1);
    Tensor out_pool2 = pool2.forward(out_conv2);
    Matrice out_flat = flatten.forward(out_pool2);
    Matrice out_dense1 = dense1.forward(out_flat);
    Matrice out_relu = relu.forward(out_dense1);
    Matrice logits = dense_out.forward(out_relu);

    CategoricalCrossEntropyWithSoftmax loss_fn;
    double total_loss = loss_fn.forward(logits, Y_test);
    Matrice probs = loss_fn.probs;

    int correct = 0;
    for (int i = 0; i < total_test; i++) {
        int best_class = 0;
        double best_prob = probs.contenu[i][0];
        for (int c = 1; c < 10; c++) {
            if (probs.contenu[i][c] > best_prob) {
                best_prob = probs.contenu[i][c];
                best_class = c;
            }
        }
        if (best_class == Y_test[i]) correct++;
    }

    double accuracy = (static_cast<double>(correct) / total_test) * 100.0;
    std::cout << "Nombre d'images de test : " << total_test << std::endl;
    std::cout << "Predictions exactes    : " << correct << " / " << total_test << std::endl;
    std::cout << "Precision (Accuracy)   : " << std::fixed << std::setprecision(2) << accuracy << " %" << std::endl;
    std::cout << "Perte (Loss) sur Test  : " << std::setprecision(6) << total_loss << std::endl;
    std::cout << "========================================\n" << std::endl;
}

// ─────────────────────────────────────────────────────────────
// Evaluation du LSTM sur le segment temporel de test (20% finaux)
// ─────────────────────────────────────────────────────────────
void tester_modele_lstm(const Par_LSTM_thread& modele, const Sequency& seq_data, 
                        int input_size, int hidden_size, int seq_len, int target_col) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "       EVALUATION DU MODELE LSTM        " << std::endl;
    std::cout << "========================================" << std::endl;

    int total_timesteps = seq_data.total_samples;
    int max_sequences = total_timesteps - seq_len - 1;
    if (max_sequences <= 0) {
        std::cerr << "Erreur : donnees insuffisantes pour l'evaluation." << std::endl;
        return;
    }

    int train_size = static_cast<int>(max_sequences * 0.80);
    int test_sequences = max_sequences - train_size;

    double total_mse = 0.0;
    double total_mae = 0.0;
    int total_predictions = 0;

    for (int i = train_size; i < max_sequences; i++) {
        Matrice h_t = zeros(hidden_size, 1);
        Matrice c_t = zeros(hidden_size, 1);

        for (int t = 0; t < seq_len; t++) {
            Matrice c_prev = c_t;

            Matrice x_t = zeros(input_size, 1);
            for (int k = 0; k < input_size; k++) {
                x_t.contenu[k][0] = seq_data.data[i + t][k];
            }
            double target = seq_data.data[i + t + 1][target_col];

            Matrice x_concat;
            concatLigne(&x_concat, x_t);
            concatLigne(&x_concat, h_t);

            Matrice a_ih = productmatriciel(modele.W_ih, x_concat);
            for (int r = 0; r < 4 * hidden_size; r++) {
                a_ih.contenu[r][0] += modele.b_ih.contenu[r][0];
            }

            Matrice f_t = zeros(hidden_size, 1);
            Matrice i_t = zeros(hidden_size, 1);
            Matrice g_t = zeros(hidden_size, 1);
            Matrice o_t = zeros(hidden_size, 1);

            for (int k = 0; k < hidden_size; k++) {
                f_t.contenu[k][0] = sigmoid(a_ih.contenu[k][0]);
                i_t.contenu[k][0] = sigmoid(a_ih.contenu[hidden_size + k][0]);
                g_t.contenu[k][0] = std::tanh(a_ih.contenu[2 * hidden_size + k][0]);
                o_t.contenu[k][0] = sigmoid(a_ih.contenu[3 * hidden_size + k][0]);
            }

            c_t = zeros(hidden_size, 1);
            h_t = zeros(hidden_size, 1);
            for (int k = 0; k < hidden_size; k++) {
                c_t.contenu[k][0] = f_t.contenu[k][0] * c_prev.contenu[k][0] + i_t.contenu[k][0] * g_t.contenu[k][0];
                h_t.contenu[k][0] = o_t.contenu[k][0] * std::tanh(c_t.contenu[k][0]);
            }

            double y_pred = modele.b_y.contenu[0][0];
            for (int k = 0; k < hidden_size; k++) {
                y_pred += modele.W_hy.contenu[0][k] * h_t.contenu[k][0];
            }

            double err = y_pred - target;
            total_mse += err * err;
            total_mae += std::fabs(err);
            total_predictions++;
        }
    }

    double final_mse = total_mse / total_predictions;
    double final_rmse = std::sqrt(final_mse);
    double final_mae = total_mae / total_predictions;

    std::cout << "Nombre de fenetres testees   : " << test_sequences << std::endl;
    std::cout << "Points temporels evalues     : " << total_predictions << std::endl;
    std::cout << "MSE sur test (normalisee)    : " << final_mse << std::endl;
    std::cout << "RMSE sur test (ecart-type)   : " << final_rmse << std::endl;
    std::cout << "MAE sur test                 : " << final_mae << std::endl;
    std::cout << "========================================\n" << std::endl;
}
