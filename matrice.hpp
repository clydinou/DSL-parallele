#ifndef _matrice
#define _matrice

#include <cmath>    // equivalent de math.h
#include <iostream> // Pour std::cout et std::cin
#include <cstdlib>  // equivalent de stdlib.h
#include <cstring>  // equivalent de string.h
#include <ctime>    // equivalent de time.h
#include <vector>   // Ideal pour gerer les matrices dynamiques
#include <string>   // Pour std::string
#include <fstream>  // Pour la gestion des fichiers
#include <random>

typedef struct  {
    int n;
    int m;
    std::vector<std::vector<double>> contenu;
} Matrice;

typedef struct {
    int n;          /* Batch size                              */
    int c;          /* Canaux (channels)                       */
    int h;          /* Hauteur                                 */
    int w;          /* Largeur                                 */
    /* Layout 4D natif : data[n][c][h][w] */
    std::vector<std::vector<std::vector<std::vector<double>>>> data;
} Tensor;

// fonction pour matrice
int containtM(Matrice A, double mot, int colone);
Matrice selectDistinctM(Matrice A, int colone);
Matrice inverse(Matrice A);
Matrice product(Matrice A, Matrice B);
Matrice productmatriciel(Matrice A, Matrice B);
void copie(Matrice * copi, Matrice A);
Matrice somme(Matrice A, Matrice B);
double sum(const std::vector<double>& tab);
Matrice difference(Matrice A, Matrice B);
Matrice comatrice(Matrice A);
void extract(Matrice * r, Matrice A, int n1, int n2, int m1, int m2);
Matrice sousMatrice(Matrice A, int ligne);
Matrice copieSauf(Matrice A, int ligne, int colonne);
Matrice transpose(Matrice A);
double determinant(Matrice A);
void produit_scalaire_matrice( Matrice *A, double alpha);
void somme_scalaire_matrice( Matrice *A, double alpha);
// Matrice random_matrice(int n, int m) ;
// void sous_interne(Matrice *A, Matrice B);
Matrice ones(int n, int m);
Matrice initialisation_matrice(int n, int m, double value);
// Matrice two(int n, int m);
Matrice zeros(int n, int m);
// Matrice system_lineaire(Matrice A);
void concatLigne(Matrice * initiale, Matrice A);
void extractLigne(Matrice * line, Matrice A, int l);
void extractColonne(Matrice * column, Matrice A, int c);
void concatColonne(Matrice * initiale, Matrice A);
// Matrice id(int n);
int compter_nb_ligne(std::string filename);
void afficher(Matrice A);
int random_values(int min, int max);
void deleteLine(Matrice * A, int line);
// void mixer_matrice(Matrice * A);
// void mixer_matrices(Matrice * A, Matrice * B);
void readMatrice(Matrice * A, std::string filename);
void readDatas(Matrice * A, Matrice * B, std::string filename, int ncolumn);
int contient(Matrice A, double v);
void write_in_file(Matrice A, std::string filename, const std::string sep);
// double distance(Matrice a, Matrice b);
Matrice random_matrice(int n, int m);
// void som_interne(Matrice *A, Matrice B);
Matrice normaliser_centrer(Matrice X);
Matrice moyenne(Matrice X);


// ─── fonctions pour tenseur ───────────────────────────────────────────────────

// Creation / initialisation
Tensor initialisation_tensor(int n, int c, int h, int w);
void randomize_tensor(Tensor *t, float min, float max);

// Lecture / affichage
Tensor readTensor(Tensor *A, Matrice *labels, std::string filename);
void afficherTensor(const Tensor *A);

// Forme
void shapeTensor(Tensor *A);      // affiche la forme (n,c,h,w)
void reshapeTensor(Tensor *A);    // remet data en coherence avec n,c,h,w
Tensor flattenTensor(Tensor *A);  // (n,c,h,w) → (n, c*h*w) ← Matrice 2-D comme tenseur
void addimTensor(Tensor *A);      // ajoute une dimension batch (n=1)
void sousdimTensor(Tensor *A);    // enleve la dimension batch si n==1

// Transformations
Tensor transposeTensor(Tensor *A);             // echange h et w
Tensor copieTensor(const Tensor *A);           // copie profonde
Tensor sommeTensor(Tensor A, Tensor B);        // addition element par element
Tensor differenceTensor(Tensor A, Tensor B);   // soustraction element par element
void produit_scalaire_tensor(float alpha, Tensor *A); // multiplication par un scalaire

// Conversion
Matrice tensor_to_matrice(const Tensor *A);    // aplatit le tenseur en matrice 2-D
Tensor matrice_to_tensor(const Matrice *M, int n, int c, int h, int w); // inverse

#endif