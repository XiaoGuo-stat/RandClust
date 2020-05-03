#include <Rcpp.h>
#include <R_ext/Lapack.h>

using Rcpp::NumericMatrix;

// [[Rcpp::export]]
NumericMatrix qr_Q(NumericMatrix x)
{
    NumericMatrix A = Rcpp::clone(x);
    const int m = A.nrow();
    const int n = A.ncol();
    if(m < n)
        Rcpp::stop("nrow(x) must be greater than or equal to ncol(x)");

    double* tau = new double[n];
    double work_query;
    int lwork = -1;
    int info;

    F77_CALL(dgeqrf)(&m, &n, A.begin(), &m, tau, &work_query, &lwork, &info);

    lwork = (int)(work_query);
    double* work = new double[lwork];
    F77_CALL(dgeqrf)(&m, &n, A.begin(), &m, tau, work, &lwork, &info);
    delete [] work;

    lwork = -1;
    F77_CALL(dorgqr)(&m, &n, &n, A.begin(), &m, tau, &work_query, &lwork, &info);

    lwork = (int)(work_query);
    work = new double[lwork];
    F77_CALL(dorgqr)(&m, &n, &n, A.begin(), &m, tau, work, &lwork, &info);
    delete [] work;

    delete [] tau;

    return A;
}
