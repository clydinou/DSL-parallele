#include "matrice.hpp"
#include <algorithm>
#include <stdexcept>
#include <sstream>

// ─────────────────────────────────────────────
//  Générateur aléatoire thread-safe
// ─────────────────────────────────────────────
// rand()/RAND_MAX ne sont pas garantis thread-safe : des appels concurrents
// depuis plusieurs threads (ex. constructeurs de Conv2D/Dense appelés en
// parallèle par cnn_local) constituent une data race, un comportement
// indéfini pouvant corrompre la mémoire de façon différée et aléatoire.
// On utilise donc un moteur std::mt19937 propre à CHAQUE thread (thread_local),
// sans aucun état partagé, donc sans race possible par construction.
static std::mt19937& rng_local() {
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}

// ─────────────────────────────────────────────
//  Utilitaires internes
// ─────────────────────────────────────────────

/* Crée une matrice n×m remplie de 0 */
Matrice zeros(int n, int m) {
    Matrice M;
    M.n = n;
    M.m = m;
    M.contenu.assign(n, std::vector<double>(m, 0.0));
    return M;
}

// ─────────────────────────────────────────────
//  Initialisation / remplissage
// ─────────────────────────────────────────────

Matrice initialisation_matrice(int n, int m, double value) {
    Matrice M = zeros(n, m);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            M.contenu[i][j] = value;
    return M;
}

// Matrice random_matrice(int n , int m){
//     std::radom_device rd ;
//     std::mt19937 gen(rd());

//     Matrice M = zeros(n, m);
//     for (int i = 0; i < n ; i++){ 
//         for(int j = 0 ; j < m ; j++){ 
//             std::uniform_real_distribution<double> dis(-1.0,1.0) ;
//             M.contenu[i][j] =dis(gen)
//         }
//     }
//     return M ;    
// }

Matrice ones(int n, int m) {
    return initialisation_matrice(n, m, 1.0);
}


Matrice random_matrice(int n, int m) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    Matrice M = zeros(n, m);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            M.contenu[i][j] = dist(rng_local());
    return M;
}

// ─────────────────────────────────────────────
//  Copie / extraction / concaténation
// ─────────────────────────────────────────────

double sum(const std::vector<double>& tab) {
    double total = 0;
    for (double element : tab) {
        total += element;
    }
    return total;
}

void copie(Matrice* copi, Matrice A) {
    copi->n = A.n;
    copi->m = A.m;
    copi->contenu = A.contenu;
}

/* Extrait le sous-bloc [n1..n2] × [m1..m2] (indices inclusifs, 0-based) */
void extract(Matrice* r, Matrice A, int n1, int n2, int m1, int m2) {
    r->n = n2 - n1 + 1;
    r->m = m2 - m1 + 1;
    r->contenu.assign(r->n, std::vector<double>(r->m, 0.0));
    for (int i = n1; i <= n2; ++i)
        for (int j = m1; j <= m2; ++j)
            r->contenu[i - n1][j - m1] = A.contenu[i][j];
}

void extractLigne(Matrice* line, Matrice A, int l) {
    line->n = 1;
    line->m = A.m;
    line->contenu = {A.contenu[l]};
}

void extractColonne(Matrice* column, Matrice A, int c) {
    column->n = A.n;
    column->m = 1;
    column->contenu.assign(A.n, std::vector<double>(1));
    for (int i = 0; i < A.n; ++i)
        column->contenu[i][0] = A.contenu[i][c];
}

void concatLigne(Matrice* initiale, Matrice A) {
    if (initiale->m == 0 && initiale->n == 0) {
        copie(initiale, A);
        return;
    }
    if (A.m != initiale->m)
        throw std::invalid_argument("concatLigne: nombre de colonnes incompatible");
    for (int i = 0; i < A.n; ++i)
        initiale->contenu.push_back(A.contenu[i]);
    initiale->n += A.n;
}

void concatColonne(Matrice* initiale, Matrice A) {
    if (initiale->n == 0 && initiale->m == 0) {
        copie(initiale, A);
        return;
    }
    if (A.n != initiale->n)
        throw std::invalid_argument("concatColonne: nombre de lignes incompatible");
    for (int i = 0; i < initiale->n; ++i)
        for (int j = 0; j < A.m; ++j)
            initiale->contenu[i].push_back(A.contenu[i][j]);
    initiale->m += A.m;
}

void deleteLine(Matrice* A, int line) {
    if (line < 0 || line >= A->n)
        throw std::out_of_range("deleteLine: indice hors bornes");
    A->contenu.erase(A->contenu.begin() + line);
    A->n -= 1;
}

// ─────────────────────────────────────────────
//  Opérations algébriques de base
// ─────────────────────────────────────────────

Matrice somme(Matrice A, Matrice B) {
    if (A.m != B.m)
        throw std::invalid_argument("somme: dimensions incompatibles");
    Matrice C = zeros(A.n, A.m);
    for (int i = 0; i < A.n; ++i)
        for (int j = 0; j < A.m; ++j)
            C.contenu[i][j] = A.contenu[i][j] + B.contenu[0][j];
    return C;
}

Matrice difference(Matrice A, Matrice B) {
    if (A.n != B.n || A.m != B.m)
        throw std::invalid_argument("difference: dimensions incompatibles");
    Matrice C = zeros(A.n, A.m);
    for (int i = 0; i < A.n; ++i)
        for (int j = 0; j < A.m; ++j)
            C.contenu[i][j] = A.contenu[i][j] - B.contenu[i][j];
    return C;
}

Matrice product(Matrice A, Matrice B) {
    if (A.m != B.m)
        throw std::invalid_argument("product: dimensions incompatibles (A.m != B.m)");
    if (A.n != B.n)
        throw std::invalid_argument("product: dimensions incompatibles (A.n != B.n)");
    Matrice C = zeros(A.n, B.m);
    for (int i = 0; i < A.n; ++i)
        for (int j = 0; j < B.m; ++j)
            C.contenu[i][j] = A.contenu[i][j] * B.contenu[i][j];
    return C;
}

Matrice productmatriciel(Matrice A, Matrice B) {
    if (A.m != B.n)
        throw std::invalid_argument("product: dimensions incompatibles (A.m != B.n)");
    Matrice C = zeros(A.n, B.m);
    for (int i = 0; i < A.n; ++i)
        for (int k = 0; k < A.m; ++k)
            for (int j = 0; j < B.m; ++j)
                C.contenu[i][j] += A.contenu[i][k] * B.contenu[k][j];
    return C;
}

void produit_scalaire_matrice(Matrice* A, double alpha) {
    for (int i = 0; i < A->n; ++i)
        for (int j = 0; j < A->m; ++j)
            A->contenu[i][j] *= alpha;
}

void somme_scalaire_matrice(Matrice* A, double alpha) {
    for (int i = 0; i < A->n; ++i)
        for (int j = 0; j < A->m; ++j)
            A->contenu[i][j] += alpha;
}

Matrice transpose(Matrice A) {
    Matrice T = zeros(A.m, A.n);
    for (int i = 0; i < A.n; ++i)
        for (int j = 0; j < A.m; ++j)
            T.contenu[j][i] = A.contenu[i][j];
    return T;
}

// ─────────────────────────────────────────────
//  Déterminant, comatrice, inverse
// ─────────────────────────────────────────────

/* Retourne la sous-matrice obtenue en supprimant la ligne `ligne` et la
   colonne `colonne` (0-based). */
Matrice copieSauf(Matrice A, int ligne, int colonne) {
    int n = A.n - 1;
    int m = A.m - 1;
    Matrice R = zeros(n, m);
    int ri = 0;
    for (int i = 0; i < A.n; ++i) {
        if (i == ligne) continue;
        int rj = 0;
        for (int j = 0; j < A.m; ++j) {
            if (j == colonne) continue;
            R.contenu[ri][rj] = A.contenu[i][j];
            ++rj;
        }
        ++ri;
    }
    return R;
}

/* Retourne la sous-matrice obtenue en supprimant uniquement la ligne `ligne`
   (toutes les colonnes conservées). Utilisée p.ex. dans certaines routines
   de sélection. */
Matrice sousMatrice(Matrice A, int ligne) {
    Matrice R = zeros(A.n - 1, A.m);
    int ri = 0;
    for (int i = 0; i < A.n; ++i) {
        if (i == ligne) continue;
        R.contenu[ri] = A.contenu[i];
        ++ri;
    }
    return R;
}

double determinant(Matrice A) {
    if (A.n != A.m)
        throw std::invalid_argument("determinant: matrice non carree");
    int n = A.n;
    if (n == 1) return A.contenu[0][0];
    if (n == 2)
        return A.contenu[0][0] * A.contenu[1][1]
             - A.contenu[0][1] * A.contenu[1][0];
    double det = 0.0;
    for (int j = 0; j < n; ++j) {
        double sign = (j % 2 == 0) ? 1.0 : -1.0;
        det += sign * A.contenu[0][j] * determinant(copieSauf(A, 0, j));
    }
    return det;
}

Matrice comatrice(Matrice A) {
    if (A.n != A.m)
        throw std::invalid_argument("comatrice: matrice non carree");
    int n = A.n;
    Matrice C = zeros(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
            C.contenu[i][j] = sign * determinant(copieSauf(A, i, j));
        }
    return C;
}

Matrice inverse(Matrice A) {
    double det = determinant(A);
    if (std::fabs(det) < 1e-15)
        throw std::runtime_error("inverse: matrice singuliere (det ≈ 0)");
    Matrice com = comatrice(A);
    Matrice tcom = transpose(com);
    produit_scalaire_matrice(&tcom, 1.0 / det);
    return tcom;
}

// ─────────────────────────────────────────────
//  Statistiques
// ─────────────────────────────────────────────

/* Moyenne par colonne : retourne une matrice 1×m */
Matrice moyenne(Matrice X) {
    Matrice M = zeros(1, X.m);
    for (int j = 0; j < X.m; ++j) {
        double s = 0.0;
        for (int i = 0; i < X.n; ++i)
            s += X.contenu[i][j];
        M.contenu[0][j] = s / X.n;
    }
    return M;
}

/* Normalisation centrée-réduite colonne par colonne */
Matrice normaliser_centrer(Matrice X) {
    Matrice moy = moyenne(X);
    Matrice R = zeros(X.n, X.m);
    // Calcul des écarts-types
    std::vector<double> sigma(X.m, 0.0);
    for (int j = 0; j < X.m; ++j) {
        double var = 0.0;
        for (int i = 0; i < X.n; ++i) {
            double d = X.contenu[i][j] - moy.contenu[0][j];
            var += d * d;
        }
        sigma[j] = sqrt(var / X.n);
        if (sigma[j] < 1e-15) sigma[j] = 1.0; // évite la division par 0
    }
    for (int i = 0; i < X.n; ++i)
        for (int j = 0; j < X.m; ++j)
            R.contenu[i][j] = (X.contenu[i][j] - moy.contenu[0][j]) / sigma[j];
    return R;
}

// ─────────────────────────────────────────────
//  Recherche / sélection
// ─────────────────────────────────────────────

/* Retourne 1 si `mot` est présent dans la colonne `colone` de A, 0 sinon */
int containtM(Matrice A, double mot, int colone) {
    for (int i = 0; i < A.n; ++i)
        if (A.contenu[i][colone] == mot) return 1;
    return 0;
}

/* Retourne 1 si `v` est présent n'importe où dans A, 0 sinon */
int contient(Matrice A, double v) {
    for (int i = 0; i < A.n; ++i)
        for (int j = 0; j < A.m; ++j)
            if (A.contenu[i][j] == v) return 1;
    return 0;
}

/* Retourne une matrice contenant les lignes de A ayant des valeurs distinctes
   dans la colonne `colone`. */
Matrice selectDistinctM(Matrice A, int colone) {
    Matrice R;
    R.n = 0;
    R.m = A.m;
    for (int i = 0; i < A.n; ++i) {
        double val = A.contenu[i][colone];
        if (!containtM(R, val, colone)) {
            R.contenu.push_back(A.contenu[i]);
            R.n += 1;
        }
    }
    return R;
}

// ─────────────────────────────────────────────
//  Affichage
// ─────────────────────────────────────────────

void afficher(Matrice A) {
    std::cout << "Matrice (" << A.n << " x " << A.m << ")\n";
    for (int i = 0; i < A.n; ++i) {
        for (int j = 0; j < A.m; ++j) {
            std::cout << A.contenu[i][j];
            if (j < A.m - 1) std::cout << "\t";
        }
        std::cout << "\n";
    }
}

// ─────────────────────────────────────────────
//  E/S fichiers
// ─────────────────────────────────────────────

/* Compte le nombre de lignes non vides dans un fichier */
int compter_nb_ligne(std::string filename) {
    std::ifstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("compter_nb_ligne: impossible d'ouvrir " + filename);
    int count = 0;
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) ++count;
    return count;
}

/* Lit un fichier CSV/TSV (séparateur espace, virgule ou tabulation) dans A */
void readMatrice(Matrice* A, std::string filename) {
    std::ifstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("readMatrice: impossible d'ouvrir " + filename);
    A->contenu.clear();
    A->n = 0;
    A->m = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        // Remplace les virgules et tabulations par des espaces
        std::replace(line.begin(), line.end(), ',', ' ');
        std::replace(line.begin(), line.end(), '\t', ' ');
        std::istringstream iss(line);
        std::vector<double> row;
        double val;
        while (iss >> val)
            row.push_back(val);
        if (row.empty()) continue;
        if (A->m == 0) A->m = (int)row.size();
        A->contenu.push_back(row);
        A->n += 1;
    }
}

/* Lit un fichier et sépare les `ncolumn` premières colonnes (features → A)
   des colonnes restantes (labels → B). */
void readDatas(Matrice* A, Matrice* B, std::string filename, int ncolumn) {
    Matrice full;
    readMatrice(&full, filename);

    int totalCols = full.m;
    int bcols = totalCols - ncolumn;

    A->n = full.n; A->m = ncolumn;
    B->n = full.n; B->m = bcols;
    A->contenu.assign(full.n, std::vector<double>(ncolumn));
    B->contenu.assign(full.n, std::vector<double>(bcols));

    for (int i = 0; i < full.n; ++i) {
        for (int j = 0; j < ncolumn; ++j)
            A->contenu[i][j] = full.contenu[i][j];
        for (int j = 0; j < bcols; ++j)
            B->contenu[i][j] = full.contenu[i][ncolumn + j];
    }
}

/* Écrit A dans filename avec le séparateur `sep` */
void write_in_file(Matrice A, std::string filename, const std::string sep) {
    std::ofstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("write_in_file: impossible d'ouvrir " + filename);
    for (int i = 0; i < A.n; ++i) {
        for (int j = 0; j < A.m; ++j) {
            f << A.contenu[i][j];
            if (j < A.m - 1) f << sep;
        }
        f << "\n";
    }
}

// ─────────────────────────────────────────────
//  Divers
// ─────────────────────────────────────────────

int random_values(int min, int max) {
    if (max <= min) return min;
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng_local());
}

// =============================================================================
//  TENSEURS  (layout 4D natif : data[n][c][h][w])
//  Accès à l'élément (i,k,y,x) :  data[i][k][y][x]
// =============================================================================

// ─────────────────────────────────────────────
//  Création / initialisation
// ─────────────────────────────────────────────

Tensor initialisation_tensor(int n, int c, int h, int w) {
    Tensor T;
    T.n = n; T.c = c; T.h = h; T.w = w;
    // Layout 4D : data[n][c][h][w]
    T.data.assign(n,
        std::vector<std::vector<std::vector<double>>>(
            c, std::vector<std::vector<double>>(
                h, std::vector<double>(w, 0.0))));
    return T;
}

void randomize_tensor(Tensor* t, float min_v, float max_v) {
    std::uniform_real_distribution<float> dist(min_v, max_v);
    for (int i = 0; i < t->n; ++i)
        for (int k = 0; k < t->c; ++k)
            for (int y = 0; y < t->h; ++y)
                for (int x = 0; x < t->w; ++x)
                    t->data[i][k][y][x] = (double)dist(rng_local());
}

// ─────────────────────────────────────────────
//  Lecture fichier
//  Format attendu : chaque ligne = label <espace> pixel_0 pixel_1 ...
//  Les pixels doivent remplir exactement c*h*w valeurs par image.
// ─────────────────────────────────────────────

Tensor readTensor(Tensor* A, Matrice* labels, std::string filename) {
    std::ifstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("readTensor: impossible d'ouvrir " + filename);

    int pixelsPerImage = A->c * A->h * A->w;

    // Pré-allocation 4D : data[n][c][h][w]
    A->data.assign(A->n,
        std::vector<std::vector<std::vector<double>>>(
            A->c, std::vector<std::vector<double>>(
                A->h, std::vector<double>(A->w, 0.0))));

    // Pré-allocation de la matrice de labels
    if (labels != nullptr) {
        labels->n = A->n;
        labels->m = 1;
        labels->contenu.assign(A->n, std::vector<double>(1, 0.0));
    }

    std::string line;
    int imgIdx = 0;

    while (std::getline(f, line) && imgIdx < A->n) {
        if (line.empty()) continue;

        // Sauter la ligne d'entête si le premier caractère n'est pas un chiffre
        if (!std::isdigit(line[0])) continue;

        std::replace(line.begin(), line.end(), ',', ' ');
        std::replace(line.begin(), line.end(), '\t', ' ');
        std::istringstream iss(line);

        if (labels != nullptr) {
            double lbl = 0.0;
            iss >> lbl;
            labels->contenu[imgIdx][0] = lbl;
        }

        // Remplissage 4D : data[i][k][y][x]
        for (int p = 0; p < pixelsPerImage; ++p) {
            int k = p / (A->h * A->w);
            int rem = p % (A->h * A->w);
            int y = rem / A->w;
            int x = rem % A->w;
            double val = 0.0;
            iss >> val;
            A->data[imgIdx][k][y][x] = val / 255.0; // normalisation [0,1]
        }
        ++imgIdx;
    }

    // Ajustement aux dimensions réelles
    A->n = imgIdx;
    A->data.resize(imgIdx);
    if (labels != nullptr) {
        labels->n = imgIdx;
        labels->contenu.resize(imgIdx);
    }

    return *A;
}

// ─────────────────────────────────────────────
//  Affichage
// ─────────────────────────────────────────────

void afficherTensor(const Tensor* A) {
    std::cout << "Tensor shape: (" << A->n << ", " << A->c
              << ", " << A->h << ", " << A->w << ")\n";
    for (int i = 0; i < A->n; ++i) {
        std::cout << "  [Batch " << i << "]\n";
        for (int k = 0; k < A->c; ++k) {
            std::cout << "    [Canal " << k << "]\n";
            for (int y = 0; y < A->h; ++y) {
                std::cout << "      ";
                for (int x = 0; x < A->w; ++x) {
                    std::cout << A->data[i][k][y][x];
                    if (x < A->w - 1) std::cout << "\t";
                }
                std::cout << "\n";
            }
        }
    }
}

// ─────────────────────────────────────────────
//  Forme
// ─────────────────────────────────────────────

void shapeTensor(Tensor* A) {
    std::cout << "Tensor shape: (" << A->n << ", " << A->c
              << ", " << A->h << ", " << A->w << ")\n";
}

/* Remet data en cohérence avec les champs n,c,h,w (redimensionne sans
   perdre les données si elles tiennent). */
void reshapeTensor(Tensor* A) {
    // Redimensionne data au format 4D cohérent avec n,c,h,w
    A->data.resize(A->n);
    for (int i = 0; i < A->n; ++i) {
        A->data[i].resize(A->c);
        for (int k = 0; k < A->c; ++k) {
            A->data[i][k].resize(A->h);
            for (int y = 0; y < A->h; ++y)
                A->data[i][k][y].resize(A->w, 0.0);
        }
    }
}

/* Aplatit un tenseur 4D en tenseur 2D (n, 1, 1, c*h*w). */
Tensor flattenTensor(Tensor* A) {
    int flat = A->c * A->h * A->w;
    Tensor F = initialisation_tensor(A->n, 1, 1, flat);
    for (int i = 0; i < A->n; ++i) {
        int pos = 0;
        for (int k = 0; k < A->c; ++k)
            for (int y = 0; y < A->h; ++y)
                for (int x = 0; x < A->w; ++x)
                    F.data[i][0][0][pos++] = A->data[i][k][y][x];
    }
    return F;
}

void addimTensor(Tensor* A) {
    // (n,c,h,w) → semantically (1, n*c, h, w) — on regroupe n dans c
    Tensor T = initialisation_tensor(1, A->n * A->c, A->h, A->w);
    for (int i = 0; i < A->n; ++i)
        for (int k = 0; k < A->c; ++k)
            T.data[0][i * A->c + k] = A->data[i][k];
    *A = T;
}

void sousdimTensor(Tensor* A) {
    if (A->n != 1)
        throw std::invalid_argument("sousdimTensor: n doit valoir 1");
    // (1, c, h, w) → (c, 1, h, w)
    Tensor T = initialisation_tensor(A->c, 1, A->h, A->w);
    for (int k = 0; k < A->c; ++k)
        T.data[k][0] = A->data[0][k];
    *A = T;
}

// ─────────────────────────────────────────────
//  Transformations
// ─────────────────────────────────────────────

/* Transpose chaque plan h×w en w×h. */
Tensor transposeTensor(Tensor* A) {
    Tensor T = initialisation_tensor(A->n, A->c, A->w, A->h);
    for (int i = 0; i < A->n; ++i)
        for (int k = 0; k < A->c; ++k)
            for (int y = 0; y < A->h; ++y)
                for (int x = 0; x < A->w; ++x)
                    T.data[i][k][x][y] = A->data[i][k][y][x];
    return T;
}

Tensor copieTensor(const Tensor* A) {
    Tensor C;
    C.n = A->n; C.c = A->c; C.h = A->h; C.w = A->w;
    C.data = A->data;
    return C;
}

Tensor sommeTensor(Tensor A, Tensor B) {
    if (A.n != B.n || A.c != B.c || A.h != B.h || A.w != B.w)
        throw std::invalid_argument("sommeTensor: shapes incompatibles");
    Tensor C = initialisation_tensor(A.n, A.c, A.h, A.w);
    for (int i = 0; i < A.n; ++i)
        for (int k = 0; k < A.c; ++k)
            for (int y = 0; y < A.h; ++y)
                for (int x = 0; x < A.w; ++x)
                    C.data[i][k][y][x] = A.data[i][k][y][x] + B.data[i][k][y][x];
    return C;
}

Tensor differenceTensor(Tensor A, Tensor B) {
    if (A.n != B.n || A.c != B.c || A.h != B.h || A.w != B.w)
        throw std::invalid_argument("differenceTensor: shapes incompatibles");
    Tensor C = initialisation_tensor(A.n, A.c, A.h, A.w);
    for (int i = 0; i < A.n; ++i)
        for (int k = 0; k < A.c; ++k)
            for (int y = 0; y < A.h; ++y)
                for (int x = 0; x < A.w; ++x)
                    C.data[i][k][y][x] = A.data[i][k][y][x] - B.data[i][k][y][x];
    return C;
}

void produit_scalaire_tensor(float alpha, Tensor* A) {
    for (int i = 0; i < A->n; ++i)
        for (int k = 0; k < A->c; ++k)
            for (int y = 0; y < A->h; ++y)
                for (int x = 0; x < A->w; ++x)
                    A->data[i][k][y][x] *= (double)alpha;
}

// ─────────────────────────────────────────────
//  Conversion tenseur ↔ matrice
// ─────────────────────────────────────────────

/* Aplatit le tenseur 4D en une matrice 2-D de taille (n × c*h*w). */
Matrice tensor_to_matrice(const Tensor* A) {
    int cols = A->c * A->h * A->w;
    Matrice M = zeros(A->n, cols);
    for (int i = 0; i < A->n; ++i) {
        int pos = 0;
        for (int k = 0; k < A->c; ++k)
            for (int y = 0; y < A->h; ++y)
                for (int x = 0; x < A->w; ++x)
                    M.contenu[i][pos++] = A->data[i][k][y][x];
    }
    return M;
}

/* Reconstruit un tenseur 4D (n,c,h,w) depuis une matrice 2-D (n x c*h*w). */
Tensor matrice_to_tensor(const Matrice* M, int n, int c, int h, int w) {
    if (M->n != n || M->m != c * h * w)
        throw std::invalid_argument("matrice_to_tensor: dimensions incompatibles");
    Tensor T = initialisation_tensor(n, c, h, w);
    for (int i = 0; i < n; ++i) {
        int pos = 0;
        for (int k = 0; k < c; ++k)
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    T.data[i][k][y][x] = M->contenu[i][pos++];
    }
    return T;
}