#include <Rcpp.h>

using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::IntegerVector;

// [[Rcpp::export]]
NumericMatrix spbin_prod(IntegerVector Ai, IntegerVector Aj, NumericMatrix P)
{
    const int n = P.nrow();
    const int k = P.ncol();
    const int nnz = Ai.length();
    const int* Aiptr = Ai.begin();
    const int* Ajptr = Aj.begin();

    NumericMatrix res(n, k);
    for(int j = 0; j < k; j++)
    {
        const double* vptr = P.begin() + j * n;
        double* resptr = res.begin() + j * n;
        for(int l = 0; l < nnz; l++)
        {
            resptr[Aiptr[l]] += vptr[Ajptr[l]];
        }
    }

    return res;
}

// [[Rcpp::export]]
NumericMatrix spbin_crossprod(IntegerVector Ai, IntegerVector Aj, NumericMatrix P)
{
    const int n = P.nrow();
    const int k = P.ncol();
    const int nnz = Ai.length();
    const int* Aiptr = Ai.begin();
    const int* Ajptr = Aj.begin();

    NumericMatrix res(n, k);
    for(int j = 0; j < k; j++)
    {
        const double* vptr = P.begin() + j * n;
        double* resptr = res.begin() + j * n;
        for(int l = 0; l < nnz; l++)
        {
            resptr[Ajptr[l]] += vptr[Aiptr[l]];
        }
    }

    return res;
}
