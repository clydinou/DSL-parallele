#include "matrice.hpp"
#include <algorithm>
#include <stdexcept>
#include <sstream>

static std::mt19937& rng_local() {
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}

Matrice zeros(int n, int m) {
    Matrice M;
    M.n = n;
    M.m = m;
    M.contenu.assign(n, std::vector<double>(m, 0.0));
    return M;
}

Matrice initialisation_matrice(int n, int m, double value) {
    Matrice M = zeros(n, m);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            M.contenu[i][j] = value;
    return M;
}

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

double sum(const std::vector<double>& tab) {
    double total = 0;
    for (double element : tab) total += element;
    return total;
}

void copie(Matrice* copi, Matrice A) {
    copi->n = A.n;
    copi->m = A.m;
    copi->contenu = A.contenu;
}

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

Matrice somme(Matrice A, Matrice B) {
    if (A.m != B.m)
        throw std::invalid_argument("somme: dimensions incompatibles");
    Matrice C = zeros(A.n, A.m);
    for (int i = 0; i < A.n; ++i)
        for (int j = 0; j < A.m; ++j)
            C.contenu[i][j] = A.contenu[i][j] + B.contenu[i][j];
    return C;
}

Matrice add_vector(Matrice A, Matrice B) {
    if (A.n != B.n || A.m != B.m)
        throw std::invalid_argument("add_vector: dimensions incompatibles");
    Matrice C = zeros(A.n, A.m);
    for (int i = 0; i < A.n; ++i)
        for (int j = 0; j < A.m; ++j)
            C.contenu[i][j] = A.contenu[i][j] + B.contenu[i][j];
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
    if (A.m != B.m || A.n != B.n)
        throw std::invalid_argument("product: dimensions incompatibles");
    Matrice C = zeros(A.n, B.m);
    for (int i = 0; i < A.n; ++i)
        for (int j = 0; j < B.m; ++j)
            C.contenu[i][j] = A.contenu[i][j] * B.contenu[i][j];
    return C;
}

Matrice productmatriciel(Matrice A, Matrice B) {
    if (A.m != B.n)
        throw std::invalid_argument("productmatriciel: dimensions incompatibles");
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
    if (A.n != A.m) throw std::invalid_argument("determinant: matrice non carree");
    int n = A.n;
    if (n == 1) return A.contenu[0][0];
    if (n == 2) return A.contenu[0][0] * A.contenu[1][1] - A.contenu[0][1] * A.contenu[1][0];
    double det = 0.0;
    for (int j = 0; j < n; ++j) {
        double sign = (j % 2 == 0) ? 1.0 : -1.0;
        det += sign * A.contenu[0][j] * determinant(copieSauf(A, 0, j));
    }
    return det;
}

Matrice comatrice(Matrice A) {
    if (A.n != A.m) throw std::invalid_argument("comatrice: matrice non carree");
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
    if (std::fabs(det) < 1e-15) throw std::runtime_error("inverse: matrice singuliere");
    Matrice com = comatrice(A);
    Matrice tcom = transpose(com);
    produit_scalaire_matrice(&tcom, 1.0 / det);
    return tcom;
}

Matrice moyenne(Matrice X) {
    Matrice M = zeros(1, X.m);
    for (int j = 0; j < X.m; ++j) {
        double s = 0.0;
        for (int i = 0; i < X.n; ++i) s += X.contenu[i][j];
        M.contenu[0][j] = s / X.n;
    }
    return M;
}

Matrice normaliser_centrer(Matrice X) {
    Matrice moy = moyenne(X);
    Matrice R = zeros(X.n, X.m);
    std::vector<double> sigma(X.m, 0.0);
    for (int j = 0; j < X.m; ++j) {
        double var = 0.0;
        for (int i = 0; i < X.n; ++i) {
            double d = X.contenu[i][j] - moy.contenu[0][j];
            var += d * d;
        }
        sigma[j] = std::sqrt(var / X.n);
        if (sigma[j] < 1e-15) sigma[j] = 1.0;
    }
    for (int i = 0; i < X.n; ++i)
        for (int j = 0; j < X.m; ++j)
            R.contenu[i][j] = (X.contenu[i][j] - moy.contenu[0][j]) / sigma[j];
    return R;
}

void readMatrice(Matrice* A, std::string filename) {
    std::ifstream f(filename);
    if (!f.is_open()) throw std::runtime_error("readMatrice: impossible d'ouvrir " + filename);
    A->contenu.clear();
    A->n = 0; A->m = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::replace(line.begin(), line.end(), '\t', ' ');
        std::istringstream iss(line);
        std::vector<double> row;
        double val;
        while (iss >> val) row.push_back(val);
        if (row.empty()) continue;
        if (A->m == 0) A->m = (int)row.size();
        A->contenu.push_back(row);
        A->n += 1;
    }
}

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
        for (int j = 0; j < ncolumn; ++j) A->contenu[i][j] = full.contenu[i][j];
        for (int j = 0; j < bcols; ++j) B->contenu[i][j] = full.contenu[i][ncolumn + j];
    }
}

int random_values(int min, int max) {
    if (max <= min) return min;
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng_local());
}

Tensor initialisation_tensor(int n, int c, int h, int w) {
    Tensor T;
    T.n = n; T.c = c; T.h = h; T.w = w;
    T.data.assign(n, std::vector<std::vector<std::vector<double>>>(
            c, std::vector<std::vector<double>>(h, std::vector<double>(w, 0.0))));
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

Tensor readTensor(Tensor* A, Matrice* labels, std::string filename) {
    std::ifstream f(filename);
    if (!f.is_open()) throw std::runtime_error("readTensor: impossible d'ouvrir " + filename);
    int pixelsPerImage = A->c * A->h * A->w;
    A->data.assign(A->n, std::vector<std::vector<std::vector<double>>>(
            A->c, std::vector<std::vector<double>>(A->h, std::vector<double>(A->w, 0.0))));
    if (labels != nullptr) {
        labels->n = A->n; labels->m = 1;
        labels->contenu.assign(A->n, std::vector<double>(1, 0.0));
    }
    std::string line;
    int imgIdx = 0;
    while (std::getline(f, line) && imgIdx < A->n) {
        if (line.empty() || !std::isdigit(line[0])) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::replace(line.begin(), line.end(), '\t', ' ');
        std::istringstream iss(line);
        if (labels != nullptr) {
            double lbl = 0.0;
            iss >> lbl;
            labels->contenu[imgIdx][0] = lbl;
        }
        for (int p = 0; p < pixelsPerImage; p++) {
            int k = p / (A->h * A->w);
            int rem = p % (A->h * A->w);
            int y = rem / A->w;
            int x = rem % A->w;
            double val = 0.0;
            iss >> val;
            A->data[imgIdx][k][y][x] = val / 255.0;
        }
        ++imgIdx;
    }
    A->n = imgIdx;
    A->data.resize(imgIdx);
    if (labels != nullptr) {
        labels->n = imgIdx;
        labels->contenu.resize(imgIdx);
    }
    return *A;
}

Sequency readSequence(const std::string& filename, int seq_len) {
    std::ifstream f(filename);
    if (!f.is_open()) throw std::runtime_error("readSequence: impossible d'ouvrir " + filename);
    Sequency seq;
    seq.seq_len = seq_len;
    std::string line;
    // Ignorer l'en-tête CSV
    std::getline(f, line);
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        std::string token;
        std::vector<double> row;
        // Ignorer la colonne de date
        iss >> token;
        double val;
        while (iss >> val) row.push_back(val);
        if (!row.empty()) {
            if (seq.num_features == 0) seq.num_features = (int)row.size();
            seq.data.push_back(row);
        }
    }
    seq.total_samples = (int)seq.data.size();
    if (seq.total_samples < seq_len)
        throw std::runtime_error("readSequence: serie temporelle plus courte que seq_len");
    return seq;
}

void reshapeTensor(Tensor* A) {
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

void addimTensor(Tensor* A) {
    Tensor T = initialisation_tensor(1, A->n * A->c, A->h, A->w);
    for (int i = 0; i < A->n; ++i)
        for (int k = 0; k < A->c; ++k)
            T.data[0][i * A->c + k] = A->data[i][k];
    *A = T;
}

void sousdimTensor(Tensor* A) {
    if (A->n != 1) throw std::invalid_argument("sousdimTensor: n doit valoir 1");
    Tensor T = initialisation_tensor(A->c, 1, A->h, A->w);
    for (int k = 0; k < A->c; ++k) T.data[k][0] = A->data[0][k];
    *A = T;
}

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
