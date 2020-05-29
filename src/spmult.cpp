#include <RcppEigen.h>

using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::IntegerVector;

#ifdef __AVX2__
#include <immintrin.h>
inline double gather_sum(const double* val, const int* idx_start, int n)
{
    constexpr int simd_size = 4;
    const int* idx_simd_end = idx_start + (n - n % simd_size);
    const int* idx_end = idx_start + n;

    __m256d resv = _mm256_set1_pd(0.0);
    for(; idx_start < idx_simd_end; idx_start += simd_size)
    {
        __m128i idx = _mm_loadu_si128((__m128i*) idx_start);
        __m256d v = _mm256_i32gather_pd(val, idx, sizeof(double));
        resv += v;
    }

    double res = 0.0;
    for(; idx_start < idx_end; idx_start++)
        res += val[*idx_start];

    return res + resv[0] + resv[1] + resv[2] + resv[3];
}
#else
inline double gather_sum(const double* val, const int* idx_start, int n)
{
    double res = 0.0;
    const int* idx_end = idx_start + n;
    for(; idx_start < idx_end; idx_start++)
        res += val[*idx_start];
    return res;
}
#endif



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

public:
    // mat is a binary sparse matrix of class dgCMatrix
    BinSpMat(Rcpp::S4 mat) :
        m_sp(Rcpp::as<MapSpMat>(mat)),
        m_n(m_sp.rows()), m_nnz(m_sp.nonZeros())
    {}

    // res = A * v
    void prod(const double* v, double* res) const
    {
        std::fill(res, res + m_n, 0.0);

        const int* inner = m_sp.innerIndexPtr();
        const int* outer = m_sp.outerIndexPtr();
        for(int j = 0; j < m_n; j++)
        {
            const int* Ai_start = inner + outer[j];
            const int* Ai_end = inner + outer[j + 1];
            const double rhs = v[j];
            for(; Ai_start < Ai_end; Ai_start++)
            {
                res[*Ai_start] += rhs;
            }
        }
    }

    // res = A' * v
    void tprod(const double* v, double* res) const
    {
        const int* inner = m_sp.innerIndexPtr();
        const int* outer = m_sp.outerIndexPtr();
        for(int j = 0; j < m_n; j++)
        {
            const int* Ai_start = inner + outer[j];
            res[j] = gather_sum(v, Ai_start, outer[j + 1] - outer[j]);
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
