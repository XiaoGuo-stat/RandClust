#include <RcppEigen.h>
#include "parallel_radix_sort.h"

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
    const bool m_compute_transpose;
    Coords m_At_inner;
    Coords m_At_outer;

public:
    // mat is a binary sparse matrix of class dgCMatrix
    BinSpMat(Rcpp::S4 mat, int nthread = 1) :
        m_sp(Rcpp::as<MapSpMat>(mat)),
        m_n(m_sp.rows()), m_nnz(m_sp.nonZeros()),
        m_compute_transpose(m_nnz > 10 * m_n)
    {
        // If the matrix is "dense", then compute the transpose for faster
        // matrix-vector multiplication
        if(!m_compute_transpose)
            return;

        m_At_inner.resize(m_nnz);
        m_At_outer.resize(m_n + 1);
        // Make a copy of the row indices of A
        int* Ai = new int[m_nnz];
        std::copy(m_sp.innerIndexPtr(), m_sp.innerIndexPtr() + m_nnz, Ai);
        // Expand column indices
        const int* outer = m_sp.outerIndexPtr();
        int* Aj = &m_At_inner[0];
        for(int k = 0; k < m_n; k++)
        {
            const int len = outer[k + 1] - outer[k];
            std::fill(Aj, Aj + len, k);
            Aj += len;
        }
        // Sort row indices in ascending order, and arrange column indices accordingly
        parallel_radix_sort::SortPairs(Ai, &m_At_inner[0], m_nnz, nthread);

        // Compute outer indices of A'
        int start = 0, row = 0;
        for(int l = 0; l < m_nnz - 1; l++)
        {
            row = Ai[l];
            if(Ai[l + 1] > row)
            {
                m_At_outer[row + 1] = l + 1 - start;
                start = l + 1;
            }
        }
        m_At_outer[Ai[start] + 1] = m_nnz - start;
        for(int i = 1; i < m_n; i++)
            m_At_outer[i] += m_At_outer[i - 1];
        m_At_outer[m_n] = m_nnz;

        delete[] Ai;
    }

    // res = A * v
    void prod(const double* v, double* res) const
    {
        if(m_compute_transpose)
        {
            const int* inner = &m_At_inner[0];
            const int* outer = &m_At_outer[0];
            for(int i = 0; i < m_n; i++)
            {
                const int* Aj_start = inner + outer[i];
                res[i] = gather_sum(v, Aj_start, outer[i + 1] - outer[i]);
            }
            return;
        }

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
SEXP sparse_matrix_coords(Rcpp::S4 mat, int nthread = 1)
{
    BinSpMat* sp = new BinSpMat(mat, nthread);
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
