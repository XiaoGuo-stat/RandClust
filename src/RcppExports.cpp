// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

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
// spbin_prod
NumericMatrix spbin_prod(IntegerVector Ai, IntegerVector Aj, NumericMatrix P);
RcppExport SEXP _RandClust_spbin_prod(SEXP AiSEXP, SEXP AjSEXP, SEXP PSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< IntegerVector >::type Ai(AiSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type Aj(AjSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type P(PSEXP);
    rcpp_result_gen = Rcpp::wrap(spbin_prod(Ai, Aj, P));
    return rcpp_result_gen;
END_RCPP
}
// spbin_crossprod
NumericMatrix spbin_crossprod(IntegerVector Ai, IntegerVector Aj, NumericMatrix P);
RcppExport SEXP _RandClust_spbin_crossprod(SEXP AiSEXP, SEXP AjSEXP, SEXP PSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< IntegerVector >::type Ai(AiSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type Aj(AjSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type P(PSEXP);
    rcpp_result_gen = Rcpp::wrap(spbin_crossprod(Ai, Aj, P));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_RandClust_qr_Q", (DL_FUNC) &_RandClust_qr_Q, 1},
    {"_RandClust_rsample", (DL_FUNC) &_RandClust_rsample, 2},
    {"_RandClust_spbin_prod", (DL_FUNC) &_RandClust_spbin_prod, 3},
    {"_RandClust_spbin_crossprod", (DL_FUNC) &_RandClust_spbin_crossprod, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_RandClust(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
