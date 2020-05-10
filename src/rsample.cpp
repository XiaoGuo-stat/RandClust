#include <RcppEigen.h>

using Rcpp::NumericVector;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Map<SpMat> MapSpMat;

//' Sample a sparse matrix
//'
//' @param A A sparse matrix of type "dgCMatrix".
//' @param P The probability that each edge is kept.
//'
//' @return A binary sparse matrix of type "dgCMatrix".
//' @examples library(Matrix)
//' set.seed(123)
//' n = 20
//' A = matrix(rbinom(n^2, 1, 0.5), 20, 20)
//' diag(A) = 0
//' A = as(A, "dgCMatrix")
//' A
//' rsample(A, 0.5)
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

//' Sample a symmetric sparse matrix
//'
//' @param A         A sparse matrix of type "dgCMatrix". \code{A} does not need
//'                  to be symmetric, see the parameter \code{use_lower}.
//' @param P         The probability that each edge is kept.
//' @param use_lower If \code{TRUE}/\code{FALSE}, only the lower/upper triangular
//'                  part of \code{A} is used for sampling.
//'
//' @return A lower triangular, binary, and sparse matrix of type "dgCMatrix".
//'         The diagonal elements are all zeros.
//' @examples library(Matrix)
//' set.seed(123)
//' n = 20
//' A = matrix(rbinom(n^2, 1, 0.5), 20, 20)
//' A = as(A, "dgCMatrix")
//' A
//' rsample_sym(A, 0.5, use_lower = TRUE)
//' rsample_sym(A, 0.5, use_lower = FALSE)
// [[Rcpp::export]]
Rcpp::S4 rsample_sym(Rcpp::S4 A, double P, bool use_lower = true)
{
    SpMat A2;
    if(use_lower)
        A2 = Rcpp::as<SpMat>(A).triangularView<Eigen::StrictlyLower>();
    else
        A2 = Rcpp::as<SpMat>(A).triangularView<Eigen::StrictlyUpper>().transpose();
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
