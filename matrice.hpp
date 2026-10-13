#ifndef _matrice
#define _matrice

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <stdexcept>

typedef struct Matrice {
    int n = 0;
    int m = 0;
    std::vector<std::vector<double>> contenu;
} Matrice;

typedef struct Tensor {
    int n = 0;          
    int c = 0;          
    int h = 0;          
    int w = 0;          
    std::vector<std::vector<std::vector<std::vector<double>>>> data;
} Tensor;

typedef struct Sequency {
    int total_samples = 0; 
    int seq_len = 0;       
    int num_features = 0;  
    std::vector<std::vector<double>> data; 
} Sequency;

Sequency readSequence(const std::string& filename, int seq_len = 20);

// Fonctions pour matrice
int containtM(Matrice A, double mot, int colone);
Matrice selectDistinctM(Matrice A, int colone);
Matrice inverse(Matrice A);
Matrice product(Matrice A, Matrice B);
Matrice productmatriciel(Matrice A, Matrice B);
void copie(Matrice* copi, Matrice A);
Matrice somme(Matrice A, Matrice B);
Matrice add_vector(Matrice A, Matrice B); // Addition element par element stricte
double sum(const std::vector<double>& tab);
Matrice difference(Matrice A, Matrice B);
Matrice comatrice(Matrice A);
void extract(Matrice* r, Matrice A, int n1, int n2, int m1, int m2);
Matrice sousMatrice(Matrice A, int ligne);
Matrice copieSauf(Matrice A, int ligne, int colonne);
Matrice transpose(Matrice A);
double determinant(Matrice A);
void produit_scalaire_matrice(Matrice* A, double alpha);
void somme_scalaire_matrice(Matrice* A, double alpha);
Matrice ones(int n, int m);
Matrice initialisation_matrice(int n, int m, double value);
Matrice zeros(int n, int m);
void concatLigne(Matrice* initiale, Matrice A);
void extractLigne(Matrice* line, Matrice A, int l);
void extractColonne(Matrice* column, Matrice A, int c);
void concatColonne(Matrice* initiale, Matrice A);
int compter_nb_ligne(std::string filename);
void afficher(Matrice A);
int random_values(int min, int max);
void deleteLine(Matrice* A, int line);
void readMatrice(Matrice* A, std::string filename);
void readDatas(Matrice* A, Matrice* B, std::string filename, int ncolumn);
int contient(Matrice A, double v);
void write_in_file(Matrice A, std::string filename, const std::string sep);
Matrice random_matrice(int n, int m);
Matrice normaliser_centrer(Matrice X);
Matrice moyenne(Matrice X);

// Fonctions pour tenseur
Tensor initialisation_tensor(int n, int c, int h, int w);
void randomize_tensor(Tensor* t, float min, float max);
Tensor readTensor(Tensor* A, Matrice* labels, std::string filename);
void afficherTensor(const Tensor* A);
void shapeTensor(Tensor* A);
void reshapeTensor(Tensor* A);
Tensor flattenTensor(Tensor* A);
void addimTensor(Tensor* A);
void sousdimTensor(Tensor* A);
Tensor transposeTensor(Tensor* A);
Tensor copieTensor(const Tensor* A);
Tensor sommeTensor(Tensor A, Tensor B);
Tensor differenceTensor(Tensor A, Tensor B);
void produit_scalaire_tensor(float alpha, Tensor* A);
Matrice tensor_to_matrice(const Tensor* A);
Tensor matrice_to_tensor(const Matrice* M, int n, int c, int h, int w);

#endif
