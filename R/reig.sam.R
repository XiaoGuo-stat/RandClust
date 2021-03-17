#' Compute randomized eigenvalue decomposition of a matrix using random sampling
#'
#' Compute the randomized eigenvalue decomposition of a matrix by random sampling. The randomized
#' eigen vectors and eigen values are computed. Can deal with very large
#' data matrix.
#'
#' This function computes the randomized eigenvalue decomposition of a data matrix using the random
#' sampling scheme. The data matrix \code{A} is first sampled to obtain a sparsified matrix.
#' An iterative algorithm (\code{\link[RSpectra]{svds}}) for computing the leading eigen vectors is then performed on the
#' sparsified matrix to obtain the randomized eigen vectors and eigen values.
#'
#'
#' @param A An input sparse data matrix of type "dgCMatrix". \code{A} is not necessarily a symmetric matrix, see the parameter \code{use_lower}.
#' @param P The sampling probability. Should be between 0 and 1.
#' @param use_lower If \code{TRUE/FALSE}, only the lower/upper triangular part of \code{A} is used for sampling and the following eigendecomposition steps.
#' @param k Number of eigen values requested.
#' @param tol Precision parameter of the iterative algorithm. Default is 1e-5.
#' @param ... Additional arguments of function \code{\link[RSpectra]{svds}}.

#' @return \item{vectors}{The randomized \code{k} eigen vectors.}
#'         \item{values}{The \code{k} eigen values.}
#'         \item{sparA}{The sparsified data matrix obtained by \code{rsample_sym(A,P)}.}
#'
#' @seealso \code{\link[RandClust]{rsample_sym}}, \code{\link[RSpectra]{svds}}.
#'
#'
#' @examples
#'
#' n <- 100
#' k <- 2
#' clustertrue <- rep(1:k, each = n/k)
#' A <- matrix(0, n, n)
#' for(i in 1:(n-1)) {
#'     for(j in (i+1):n) {
#'         A[i, j] <- ifelse(clustertrue[i] == clustertrue[j], rbinom(1, 1, 0.2), rbinom(1, 1, 0.1))
#'         A[j, i] <- A[i, j]
#'     }
#' }
#' diag(A) <- 0
#' A <- as(A, "dgCMatrix")
#' reig.sam(A, P = 0.7, use_lower = TRUE, k = k)
#'
#' @export reig.sam
#'
#'
reig.sam <- function(A, P, use_lower = TRUE, k, tol = 1e-5, ...){
  #Obtain the sparsified matrix
  rA <- rsample_sym (A, P, use_lower = use_lower)/P

  #Find the leading eigen vectors of the sparsified matrix
  partialeig <- svds (rA, k = k, opts = list(tol = tol), ...)
  vectors <- partialeig$u
  values <- partialeig$d

  #Output the result
  list(vectors = vectors, values = values, sparA = rA)
}



