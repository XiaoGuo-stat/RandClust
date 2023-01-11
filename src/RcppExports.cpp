// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// qr_Q
NumericMatrix qr_Q(NumericMatrix x);
RcppExport SEXP _RandClust_qr_Q(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(qr_Q(x));
    return rcpp_result_gen;
END_RCPP
}
// qr_Q2
List qr_Q2(NumericMatrix x1, NumericMatrix x2, int nthread);
RcppExport SEXP _RandClust_qr_Q2(SEXP x1SEXP, SEXP x2SEXP, SEXP nthreadSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type x1(x1SEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type x2(x2SEXP);
    Rcpp::traits::input_parameter< int >::type nthread(nthreadSEXP);
    rcpp_result_gen = Rcpp::wrap(qr_Q2(x1, x2, nthread));
    return rcpp_result_gen;
END_RCPP
}
// qr_Q_inplace
void qr_Q_inplace(NumericMatrix x);
RcppExport SEXP _RandClust_qr_Q_inplace(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type x(xSEXP);
    qr_Q_inplace(x);
    return R_NilValue;
END_RCPP
}
// qr_Q2_inplace
void qr_Q2_inplace(NumericMatrix x1, NumericMatrix x2, int nthread);
RcppExport SEXP _RandClust_qr_Q2_inplace(SEXP x1SEXP, SEXP x2SEXP, SEXP nthreadSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type x1(x1SEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type x2(x2SEXP);
    Rcpp::traits::input_parameter< int >::type nthread(nthreadSEXP);
    qr_Q2_inplace(x1, x2, nthread);
    return R_NilValue;
END_RCPP
}
// rsample
Rcpp::S4 rsample(Rcpp::S4 A, double P);
RcppExport SEXP _RandClust_rsample(SEXP ASEXP, SEXP PSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::S4 >::type A(ASEXP);
    Rcpp::traits::input_parameter< double >::type P(PSEXP);
    rcpp_result_gen = Rcpp::wrap(rsample(A, P));
    return rcpp_result_gen;
END_RCPP
}
// rsample_sym
Rcpp::S4 rsample_sym(Rcpp::S4 A, double P, bool use_lower);
RcppExport SEXP _RandClust_rsample_sym(SEXP ASEXP, SEXP PSEXP, SEXP use_lowerSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::S4 >::type A(ASEXP);
    Rcpp::traits::input_parameter< double >::type P(PSEXP);
    Rcpp::traits::input_parameter< bool >::type use_lower(use_lowerSEXP);
    rcpp_result_gen = Rcpp::wrap(rsample_sym(A, P, use_lower));
    return rcpp_result_gen;
END_RCPP
}
// sparse_matrix_coords
SEXP sparse_matrix_coords(Rcpp::S4 mat, int nthread);
RcppExport SEXP _RandClust_sparse_matrix_coords(SEXP matSEXP, SEXP nthreadSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::S4 >::type mat(matSEXP);
    Rcpp::traits::input_parameter< int >::type nthread(nthreadSEXP);
    rcpp_result_gen = Rcpp::wrap(sparse_matrix_coords(mat, nthread));
    return rcpp_result_gen;
END_RCPP
}
// spbin_power_prod
NumericMatrix spbin_power_prod(SEXP coords, NumericMatrix P, int q, int nthread);
RcppExport SEXP _RandClust_spbin_power_prod(SEXP coordsSEXP, SEXP PSEXP, SEXP qSEXP, SEXP nthreadSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type P(PSEXP);
    Rcpp::traits::input_parameter< int >::type q(qSEXP);
    Rcpp::traits::input_parameter< int >::type nthread(nthreadSEXP);
    rcpp_result_gen = Rcpp::wrap(spbin_power_prod(coords, P, q, nthread));
    return rcpp_result_gen;
END_RCPP
}
// spbin_krylov_space
NumericMatrix spbin_krylov_space(SEXP coords, NumericMatrix P, int q, int nthread);
RcppExport SEXP _RandClust_spbin_krylov_space(SEXP coordsSEXP, SEXP PSEXP, SEXP qSEXP, SEXP nthreadSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type P(PSEXP);
    Rcpp::traits::input_parameter< int >::type q(qSEXP);
    Rcpp::traits::input_parameter< int >::type nthread(nthreadSEXP);
    rcpp_result_gen = Rcpp::wrap(spbin_krylov_space(coords, P, q, nthread));
    return rcpp_result_gen;
END_RCPP
}
// spbin_power_crossprod
NumericMatrix spbin_power_crossprod(SEXP coords, NumericMatrix P, int q, int nthread);
RcppExport SEXP _RandClust_spbin_power_crossprod(SEXP coordsSEXP, SEXP PSEXP, SEXP qSEXP, SEXP nthreadSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type P(PSEXP);
    Rcpp::traits::input_parameter< int >::type q(qSEXP);
    Rcpp::traits::input_parameter< int >::type nthread(nthreadSEXP);
    rcpp_result_gen = Rcpp::wrap(spbin_power_crossprod(coords, P, q, nthread));
    return rcpp_result_gen;
END_RCPP
}
// spbin_power_crossprod_inplace
void spbin_power_crossprod_inplace(SEXP coords, NumericMatrix P, NumericMatrix res, int q, int nthread);
RcppExport SEXP _RandClust_spbin_power_crossprod_inplace(SEXP coordsSEXP, SEXP PSEXP, SEXP resSEXP, SEXP qSEXP, SEXP nthreadSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type P(PSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type res(resSEXP);
    Rcpp::traits::input_parameter< int >::type q(qSEXP);
    Rcpp::traits::input_parameter< int >::type nthread(nthreadSEXP);
    spbin_power_crossprod_inplace(coords, P, res, q, nthread);
    return R_NilValue;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_RandClust_qr_Q", (DL_FUNC) &_RandClust_qr_Q, 1},
    {"_RandClust_qr_Q2", (DL_FUNC) &_RandClust_qr_Q2, 3},
    {"_RandClust_qr_Q_inplace", (DL_FUNC) &_RandClust_qr_Q_inplace, 1},
    {"_RandClust_qr_Q2_inplace", (DL_FUNC) &_RandClust_qr_Q2_inplace, 3},
    {"_RandClust_rsample", (DL_FUNC) &_RandClust_rsample, 2},
    {"_RandClust_rsample_sym", (DL_FUNC) &_RandClust_rsample_sym, 3},
    {"_RandClust_sparse_matrix_coords", (DL_FUNC) &_RandClust_sparse_matrix_coords, 2},
    {"_RandClust_spbin_power_prod", (DL_FUNC) &_RandClust_spbin_power_prod, 4},
    {"_RandClust_spbin_krylov_space", (DL_FUNC) &_RandClust_spbin_krylov_space, 4},
    {"_RandClust_spbin_power_crossprod", (DL_FUNC) &_RandClust_spbin_power_crossprod, 4},
    {"_RandClust_spbin_power_crossprod_inplace", (DL_FUNC) &_RandClust_spbin_power_crossprod_inplace, 5},
    {NULL, NULL, 0}
};

RcppExport void R_init_RandClust(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
