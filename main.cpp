#include "fonction.hpp"
#include <iostream>

// Variables globales de chronométrage (TimePoint chrono)
TimePoint w_time_par1, w_time_par2;
double work_time_par;

int main(int argc, char* argv[]) {
    std::string filename = "mnist_test.csv";
    if (argc > 1) {
        filename = argv[1];
    }
    int n_threads = 4;
    if (argc > 2) {
        n_threads = atoi(argv[2]);
    }

    std::cout << "==================================================" << std::endl;
    std::cout << " Test du CNN Parallèle sur : " << filename << std::endl;
    std::cout << "==================================================" << std::endl;

    double alpha = 0.01;
    double seuil = 0.01;
    int n_epoch = 100;

    try {
        CNN_Weights_Par_Thread resultats = CNNAlgorithm(alpha, seuil, n_epoch, filename, n_threads);

        std::cout << "\n--------------------------------------------------" << std::endl;
        std::cout << " Temps d'exécution : " << work_time_par << " s" << std::endl;
        std::cout << " Nombre de threads : " << resultats.threads_results.size() << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;

        for (int t = 0; t < (int)resultats.threads_results.size(); t++) {
            std::cout << "Thread [" << t << "] | Loss finale : "
                      << resultats.erreurs_threads.contenu[t][0]
                      << " | Couches Conv2D : " << resultats.threads_results[t].W_conv_local.size()
                      << " | Couches Dense : "  << resultats.threads_results[t].W_dense_local.size()
                      << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Erreur lors de l'exécution : " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}