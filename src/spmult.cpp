#include <RcppEigen.h>

using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::IntegerVector;

// Binary sparse matrix
class BinSpMat
{
private:
    using SpMat = Eigen::SparseMatrix<double>;
    using MapSpMat = Eigen::Map<SpMat>;
    using Coords = std::vector<int>;

    const MapSpMat m_sp;
    const int m_n;
    const int m_nnz;
    Coords m_Ai;
    Coords m_Aj;

public:
    // mat is a binary sparse matrix of class dgCMatrix
    BinSpMat(Rcpp::S4 mat) :
        m_sp(Rcpp::as<MapSpMat>(mat)),
        m_n(m_sp.rows()), m_nnz(m_sp.nonZeros()),
        m_Ai(m_nnz), m_Aj(m_nnz)
    {
        int i = 0;
        const int n = m_sp.cols();
        for(int k = 0; k < n; k++)
        {
            for(MapSpMat::InnerIterator it(m_sp, k); it; ++it, ++i)
            {
                m_Ai[i] = it.row();
                m_Aj[i] = k;
            }
        }
    }

    // res = A * v
    void prod(const double* v, double* res) const
    {
        std::fill(res, res + m_n, 0.0);
        const int* Ai = &m_Ai[0];
        const int* Ai_end = Ai + m_nnz;
        const int* Aj = &m_Aj[0];
        for(; Ai < Ai_end; Ai++, Aj++)
            res[*Ai] += v[*Aj];
    }

    // res = A' * v
    void tprod(const double* v, double* res) const
    {
        const int* inner = m_sp.innerIndexPtr();
        const int* outer = m_sp.outerIndexPtr();
        for(int c = 0; c < m_n; c++)
        {
            const int Aj = c;
            const int* Ai_ptr = inner + outer[c];
            const int* Ai_end = inner + outer[c + 1];
            double r = 0.0;
            for(; Ai_ptr < Ai_end; Ai_ptr++)
            {
                const int Ai = *Ai_ptr;
                r += v[Ai];
            }
            res[Aj] = r;
        }
    }
};



// [[Rcpp::export]]
SEXP sparse_matrix_coords(Rcpp::S4 mat)
{
    BinSpMat* sp = new BinSpMat(mat);
    return Rcpp::XPtr<BinSpMat>(sp, true);
}

// res = (AA')^q AP
// [[Rcpp::export]]
NumericMatrix spbin_power_prod(SEXP coords, NumericMatrix P, int q = 0, int nthread = 1)
{
    Rcpp::XPtr<BinSpMat> sp(coords);
    const int n = P.nrow();
    const int k = P.ncol();
    NumericMatrix res(Rcpp::no_init_matrix(n, k));
    const double* P_ptr = P.begin();
    double* res_ptr = res.begin();

    #pragma omp parallel for shared(P_ptr, res_ptr, sp) num_threads(nthread)
    for(int j = 0; j < k; j++)
    {
        const double* v = P_ptr + j * n;
        double* r = res_ptr + j * n;
        // res = AP
        sp->prod(v, r);

        // Allocate workspace if needed
        double* work = NULL;
        if(q > 0)
            work = new double[n];

        // Power iterations
        for(int i = 0; i < q; i++)
        {
            sp->tprod(r, work);
            sp->prod(work, r);
        }

        // Free workspace
        if(q > 0)
            delete[] work;
    }

    return res;
}

// res = (A'A)^q A'P
// [[Rcpp::export]]
NumericMatrix spbin_power_crossprod(SEXP coords, NumericMatrix P, int q = 0, int nthread = 1)
{
    Rcpp::XPtr<BinSpMat> sp(coords);
    const int n = P.nrow();
    const int k = P.ncol();
    NumericMatrix res(Rcpp::no_init_matrix(n, k));
    const double* P_ptr = P.begin();
    double* res_ptr = res.begin();

    #pragma omp parallel for shared(P_ptr, res_ptr, sp) num_threads(nthread)
    for(int j = 0; j < k; j++)
    {
        const double* v = P_ptr + j * n;
        double* r = res_ptr + j * n;
        // res = A'P
        sp->tprod(v, r);

        // Allocate workspace if needed
        double* work = NULL;
        if(q > 0)
            work = new double[n];

        // Power iterations
        for(int i = 0; i < q; i++)
        {
            sp->prod(r, work);
            sp->tprod(work, r);
        }

        // Free workspace
        if(q > 0)
            delete[] work;
    }

    return res;
}
