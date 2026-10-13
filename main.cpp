#include "fonction.hpp"
#include <iostream>
#include <string>
#include <cstdlib>

TimePoint w_time_par1, w_time_par2;
double work_time_par;

int main(int argc, char* argv[]) {
    int n = 2; // Par defaut : 2 (LSTM)
    if (argc > 3) {
        n = std::atoi(argv[3]);
    }

    std::string filename = (n == 1) ? "mnist.txt" : "Microsoft_Stock.csv";
    if (argc > 1) {
        filename = argv[1];
    }

    int n_threads = 4;
    if (argc > 2) {
        n_threads = std::atoi(argv[2]);
    }

    switch (n) {
        case 1: { 
            std::cout << "==================================================" << std::endl;
            std::cout << " Test du CNN Parallele sur : " << filename << std::endl;
            std::cout << "==================================================" << std::endl;
        
            double alpha = 0.01;
            double seuil = 0.001;
            int n_epoch = 10;
        
            try {
                Par_CNN_thread resultats = CNNAlgorithm(alpha, seuil, n_epoch, filename, n_threads);
        
                std::cout << "\n--------------------------------------------------" << std::endl;
                std::cout << " Temps d'execution : " << work_time_par << " s" << std::endl;
                std::cout << " Nombre de threads : " << resultats.threads_results.size() << std::endl;
                std::cout << "--------------------------------------------------" << std::endl;
        
                for (int t = 0; t < (int)resultats.threads_results.size(); t++) {
                    std::cout << "Thread [" << t << "] | Loss finale : "
                              << resultats.erreurs_threads.contenu[t][0]
                              << " | Couches Conv2D : " << resultats.threads_results[t].W_conv_local.size()
                              << " | Couches Dense : "  << resultats.threads_results[t].W_dense_local.size()
                              << std::endl;
                }

                // Evaluation du CNN sur un échantillon de test
                Tensor X_test;
                Matrice Y_mat;
                X_test.n = 1000; X_test.c = 1; X_test.h = 28; X_test.w = 28;
                readTensor(&X_test, &Y_mat, filename);
                std::vector<int> Y_test(X_test.n);
                for (int i = 0; i < X_test.n; i++) Y_test[i] = static_cast<int>(Y_mat.contenu[i][0]);

                tester_modele_cnn(resultats, X_test, Y_test);
            }
            catch (const std::exception& e) {
                std::cerr << "Erreur lors de l'execution du CNN : " << e.what() << std::endl;
                return EXIT_FAILURE;
            }
            break;
        }

        case 2: { 
            std::cout << "==================================================" << std::endl;
            std::cout << " Test du LSTM Parallele sur : " << filename << std::endl;
            std::cout << "==================================================" << std::endl;
        
            int input_size = 3;     // Open, High, Low
            int hidden_size = 16;   // Mémoire cachée
            int n_epochs = 20;      
            int seq_len = 20;       
            double alpha = 0.01;    
            double seuil = 1e-4;    
        
            try {
                std::cout << "Lancement de l'apprentissage parallele LSTM sur " << n_threads << " threads..." << std::endl;
        
                Par_LSTM_thread resultat = LSTMAlgorithm(alpha, seuil, n_epochs, filename, n_threads, input_size, hidden_size, seq_len);
        
                std::cout << "========================================" << std::endl;
                std::cout << "        RESULTATS DE L'ENTRAINEMENT     " << std::endl;
                std::cout << "========================================" << std::endl;
                std::cout << "Epoques executees : " << resultat.epoques_effectuees << "/" << n_epochs << std::endl;
                std::cout << "MSE finale Train  : " << resultat.cout_final << std::endl;
                std::cout << "Temps d'execution : " << resultat.temps_execution << " secondes" << std::endl;

                // Evaluation du LSTM sur les 20% temporels restants
                Sequency seq_eval = readSequence(filename, seq_len);
                Matrice raw_mat = zeros(seq_eval.total_samples, seq_eval.num_features);
                for (int i = 0; i < seq_eval.total_samples; ++i)
                    for (int j = 0; j < seq_eval.num_features; ++j)
                        raw_mat.contenu[i][j] = seq_eval.data[i][j];
                raw_mat = normaliser_centrer(raw_mat);
                for (int i = 0; i < seq_eval.total_samples; ++i)
                    for (int j = 0; j < seq_eval.num_features; ++j)
                        seq_eval.data[i][j] = raw_mat.contenu[i][j];

                tester_modele_lstm(resultat, seq_eval, input_size, hidden_size, seq_len, 3);
            }
            catch (const std::exception& e) {
                std::cerr << "Erreur lors de l'execution du LSTM : " << e.what() << std::endl;
                return EXIT_FAILURE;
            }
            break;
        }

        default:
            std::cerr << "Option invalide (" << n << "). Utilisez 1 pour CNN ou 2 pour LSTM." << std::endl;
            return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
