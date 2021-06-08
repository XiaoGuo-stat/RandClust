#include <Rcpp.h>
#include <R_ext/Lapack.h>

using Rcpp::NumericMatrix;
using Rcpp::List;

// X = QR, X[m x n], store Q in res
void qr_Q_impl(const double* x, const int m, const int n, double* res)
{
    // Copy x to res
    std::copy(x, x + m * n, res);

    double* tau = new double[n];
    double work_query;
    int lwork = -1;
    int info;
    // Query workspace size
    F77_CALL(dgeqrf)(&m, &n, res, &m, tau, &work_query, &lwork, &info);

    // QR decomposition
    lwork = (int)(work_query);
    double* work = new double[lwork];
    F77_CALL(dgeqrf)(&m, &n, res, &m, tau, work, &lwork, &info);
    delete[] work;

    // Query workspace size
    lwork = -1;
    F77_CALL(dorgqr)(&m, &n, &n, res, &m, tau, &work_query, &lwork, &info);

    // Extract Q
    lwork = (int)(work_query);
    work = new double[lwork];
    F77_CALL(dorgqr)(&m, &n, &n, res, &m, tau, work, &lwork, &info);
    delete[] work;

    delete[] tau;
}

// [[Rcpp::export]]
NumericMatrix qr_Q(NumericMatrix x)
{
    const int m = x.nrow();
    const int n = x.ncol();
    if(m < n)
        Rcpp::stop("nrow(x) must be greater than or equal to ncol(x)");

    NumericMatrix A(Rcpp::no_init_matrix(m, n));
    qr_Q_impl(x.begin(), m, n, A.begin());

    return A;
}

// Two QR decompositions in parallel
// [[Rcpp::export]]
List qr_Q2(NumericMatrix x1, NumericMatrix x2, int nthread = 2)
{
    const int m1 = x1.nrow();
    const int n1 = x1.ncol();
    if(m1 < n1)
        Rcpp::stop("nrow(x1) must be greater than or equal to ncol(x1)");

    const int m2 = x2.nrow();
    const int n2 = x2.ncol();
    if(m2 < n2)
        Rcpp::stop("nrow(x2) must be greater than or equal to ncol(x2)");

    NumericMatrix A1(Rcpp::no_init_matrix(m1, n1));
    NumericMatrix A2(Rcpp::no_init_matrix(m2, n2));

    if(nthread > 1)
        nthread = 2;

    #pragma omp parallel sections num_threads(nthread)
    {
        #pragma omp section
        qr_Q_impl(x1.begin(), m1, n1, A1.begin());

        #pragma omp section
        qr_Q_impl(x2.begin(), m2, n2, A2.begin());
    }

    return List::create(A1, A2);
}



// X = QR, X[m x n], m >= n
// Overwrite X and an additional vector tau [n] with an internal representation of Q and R
void qr_inplace_impl(double* x, double* tau, const int m, const int n)
{
    double work_query;
    int lwork = -1;
    int info;
    // Query workspace size
    F77_CALL(dgeqrf)(&m, &n, x, &m, tau, &work_query, &lwork, &info);

    // QR decomposition
    lwork = (int)(work_query);
    double* work = new double[lwork];
    F77_CALL(dgeqrf)(&m, &n, x, &m, tau, work, &lwork, &info);
    delete[] work;
}

void qr_Q_inplace_impl(double* qr, const double* tau, const int m, const int n)
{
    double work_query;
    int lwork = -1;
    int info;
    // Query workspace size
    F77_CALL(dorgqr)(&m, &n, &n, qr, &m, tau, &work_query, &lwork, &info);

    // Extract Q
    lwork = (int)(work_query);
    double* work = new double[lwork];
    F77_CALL(dorgqr)(&m, &n, &n, qr, &m, tau, work, &lwork, &info);
    delete[] work;
}

// Two QR decompositions in parallel
// [[Rcpp::export]]
void qr_Q2_inplace(NumericMatrix x1, NumericMatrix x2, int nthread = 2)
{
    const int m1 = x1.nrow();
    const int n1 = x1.ncol();
    if(m1 < n1)
        Rcpp::stop("nrow(x1) must be greater than or equal to ncol(x1)");

    const int m2 = x2.nrow();
    const int n2 = x2.ncol();
    if(m2 < n2)
        Rcpp::stop("nrow(x2) must be greater than or equal to ncol(x2)");

    double* tau1 = new double[n1];
    double* tau2 = new double[n2];

    if(nthread > 1)
        nthread = 2;

    #pragma omp parallel sections num_threads(nthread)
    {
        #pragma omp section
        {
            qr_inplace_impl(x1.begin(), tau1, m1, n1);
            qr_Q_inplace_impl(x1.begin(), tau1, m1, n1);
        }

        #pragma omp section
        {
            qr_inplace_impl(x2.begin(), tau2, m2, n2);
            qr_Q_inplace_impl(x2.begin(), tau2, m2, n2);
        }
    }

    delete[] tau1;
    delete[] tau2;
}
