#' Compute randomized SVD of a matrix using random sampling
#'
#' Compute the randomized SVD of a matrix by random sampling. The randomized
#' singular vectors and singular values are computed. Can deal with very large
#' data matrix.
#'
#' This function computes the randomized SVD of a data matrix using the random
#' sampling scheme. The data matrix \code{A} is first sampled to obtain a sparsified matrix.
#' An iterative algorithm (\code{\link[irlba]{irlba}}) for computing the leading singular vectors is then performed on the
#' sparsified matrix to obtain the randomized singular vectors and singular values.
#'
#'
#' @param A An input sparse data matrix of type "dgCMatrix". Not necessarily be the adjacency matrix of a network.
#' @param P The sampling probability. Should be between 0 and 1.
#' @param k Number of singular values requested. \code{k} should not be smaller than \code{nu} or \code{nv}. The default value is
#'          \code{max(nu,nv)}.
#' @param nu Number of left singular vectors to be computed.
#' @param nv Number of right singular vectors to be computed.
#' @param tol Precision parameter of the iterative algorithm. Default is 1e-5.
#' @param ... Additional arguments of function \code{\link[irlba]{irlba}}.

#' @return \item{u}{The randomized left \code{nu} singular vectors.} \item{v}{The randomized right \code{nv} singular vectors.}
#'         \item{d}{The \code{k} leading singular values.} \item{sparA}{The sparsified data matrix obtained by \code{rsample(A,P)}.}
#'
#' @seealso \code{\link[RandClust]{rsample}}, \code{\link[irlba]{irlba}}.
#'
#'
#' @examples
#'
#' n <- 100
#' rank <- 2
#' clustertrue.y <- rep(1:rank, each = n/rank)
#' clustertrue.z <- rep(1:rank, each = n/rank)
#' A <- matrix(0, n, n)
#' for(i in 1:n) {
#'     for(j in 1:n) {
#'         A[i, j] <- ifelse(clustertrue.y[i] == clustertrue.z[i], rbinom(1, 1, 0.2), rbinom(1, 1, 0.1))
#'     }
#' }
#' diag(A) <- 0
#' A <- as(A, "dgCMatrix")
#' rsvd.sam(A, P = 0.7, nu = rank, nv = rank)
#'
#' @export rsvd.sam
#'
#'
rsvd.sam <- function(A, P, k = max(nu, nv), nu, nv, tol = 1e-5, ...)
{
    # Obtain the sparsified matrix
    rA <- rsample(A, P)

    # Find the leading singular vectors of the sparsified matrix
    # partialsvd <- svds(rA, k = k, nu = nu, nv = nv, opts = list(ncv = 2 * k, tol = tol), ...)
    partialsvd <- irlba(rA, nu = nu, nv = nv, tol = tol, ...)
    u <- partialsvd$u
    v <- partialsvd$v
    d <- partialsvd$d

    # Output the result
    list(u = u, v = v, d = d, sparA = rA)
}
