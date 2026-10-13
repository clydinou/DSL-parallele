#ifndef _FONCTION_HPP
#define _FONCTION_HPP

#include "dsl.hpp"
#include "time_manager.hpp"
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

Par_CNN_thread CNNAlgorithm(double alpha, double seuil, int n_epoch, std::string filename, int n_threads);

Par_LSTM_thread LSTMAlgorithm(double alpha, double seuil, int n_epoch, std::string filename, 
                              int n_threads, int input_size, int hidden_size, int seq_len = 20);

void tester_modele_cnn(const Par_CNN_thread& resultat_cnn, const Tensor& X_test, const std::vector<int>& Y_test);

void tester_modele_lstm(const Par_LSTM_thread& modele, const Sequency& seq_data, 
                        int input_size, int hidden_size, int seq_len, int target_col = 3);

#endif
