#include <RcppEigen.h>

using Rcpp::NumericVector;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Map<SpMat> MapSpMat;

// [[Rcpp::export]]
Rcpp::S4 rsample(Rcpp::S4 A, double P)
{
    SpMat A2 = Rcpp::as<SpMat>(A);
    const int nnz = A2.nonZeros();
    double* xptr = A2.valuePtr();
    for(int i = 0; i < nnz; i++)
    {
        xptr[i] = double(R::unif_rand() < P);
    }

    A2.prune(0.0);
    A2.makeCompressed();

    return Rcpp::wrap(A2);
}
