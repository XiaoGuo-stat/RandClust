#include <Rcpp.h>

using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::IntegerVector;

// res = A * P, A[n x n], P[n x k]
// A is a binary sparse matrix with nonzero elements given by coordinates (Ai, Aj)
void spbin_prod(const int* Ai, const int* Aj, const int nnz,
                const double* P, const int n, const int k, double* res)
{
    #pragma omp parallel for shared(Ai, Aj, P, res)
    for(int j = 0; j < k; j++)
    {
        const double* v_ptr = P + j * n;
        double* col_ptr = res + j * n;
        for(int l = 0; l < nnz; l++)
        {
            col_ptr[Ai[l]] += v_ptr[Aj[l]];
        }
    }
}

// res = A' * P, A[n x n], P[n x k]
// A is a binary sparse matrix with nonzero elements given by coordinates (Ai, Aj)
void spbin_crossprod(const int* Ai, const int* Aj, const int nnz,
                     const double* P, const int n, const int k, double* res)
{
    #pragma omp parallel for shared(Ai, Aj, P, res)
    for(int j = 0; j < k; j++)
    {
        const double* v_ptr = P + j * n;
        double* col_ptr = res + j * n;
        for(int l = 0; l < nnz; l++)
        {
            col_ptr[Aj[l]] += v_ptr[Ai[l]];
        }
    }
}



// res = (AA')^q AP
// [[Rcpp::export]]
NumericMatrix spbin_power_prod(IntegerVector Ai, IntegerVector Aj, NumericMatrix P, int q)
{
    const int n = P.nrow();
    const int k = P.ncol();
    const int nnz = Ai.length();

    NumericMatrix res(n, k);

    // res = AP
    spbin_prod(Ai.begin(), Aj.begin(), nnz, P.begin(), n, k, res.begin());

    // Allocate workspace if needed
    double* work = NULL;
    if(q > 0)
        work = new double[n * k];

    // Power iterations
    for(int i = 0; i < q; i++)
    {
        spbin_crossprod(Ai.begin(), Aj.begin(), nnz, res.begin(), n, k, work);
        spbin_prod(Ai.begin(), Aj.begin(), nnz, work, n, k, res.begin());
    }

    // Free workspace
    if(q > 0)
        delete[] work;

    return res;
}

// res = (A'A)^q A'P
// [[Rcpp::export]]
NumericMatrix spbin_power_crossprod(IntegerVector Ai, IntegerVector Aj, NumericMatrix P, int q)
{
    const int n = P.nrow();
    const int k = P.ncol();
    const int nnz = Ai.length();

    NumericMatrix res(n, k);

    // res = A'P
    spbin_crossprod(Ai.begin(), Aj.begin(), nnz, P.begin(), n, k, res.begin());

    // Allocate workspace if needed
    double* work = NULL;
    if(q > 0)
        work = new double[n * k];

    // Power iterations
    for(int i = 0; i < q; i++)
    {
        spbin_prod(Ai.begin(), Aj.begin(), nnz, res.begin(), n, k, work);
        spbin_crossprod(Ai.begin(), Aj.begin(), nnz, work, n, k, res.begin());
    }

    // Free workspace
    if(q > 0)
        delete[] work;

    return res;
}
