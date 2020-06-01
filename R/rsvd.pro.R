#' Compute randomized SVD of a matrix using random projection
#'
#' Compute the randomized SVD of a matrix by random projection. The randomized
#' singular vectors and singular values are computed. Can deal with very large
#' data matrix.
#'
#' This function computes the randomized SVD of a data matrix using the random
#' projection scheme. The data matrix \code{A} is first compressed to a
#' smaller matrix with its columns (rows) being the linear combinations of the
#' columns (rows) of \code{A}. The classical SVD is then performed on the smaller
#' matrix. The randomized SVD of \code{A} are obtained by postprocessing.
#'
#'
#'
#'
#' @param A Input data matrix of class "\code{dgCMatrix}". Not necessarily be the adjacency matrix of a network.
#' @param rank The target rank of the low-rank decomposition.
#' @param p The oversampling parameter. It need to be a positive integer number. Default value is 10.
#' @param q The power parameter. It need to be a positive integer number. Default value is 2.
#' @param dist The distribution of the entry of the random test matrix. Can be \code{"normal"} (standard normal distribution),
#'             \code{"unif"} (uniform distribution from -1 to 1), or \code{"rademacher"} (randemacher distribution). Default
#'             is \code{"normal"}.
#' @param approA A logical variable indicating whether the approximated \code{A} is returned. Default is \code{FALSE}.
#' @param nthread Maximum number of threads for specific computations which could be implemented in parallel. Default is 1.
#'
#' @return \item{u}{The randomized left \code{rank+p} singular vectors.} \item{v}{The randomized right \code{rank+p} singular vectors.}
#'         \item{d}{The \code{rank+p} singular values.} \item{approA}{The approximated data matrix obtained by {\eqn{udv}'} if requested.}
#'
#' @references N. Halko, P.-G. Martinsson, and J. A. Tropp. (2011)
#' \emph{Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions},
#' \emph{SIAM review, Vol. 53(2), 217-288}\cr
#' \url{https://epubs.siam.org/doi/10.1137/090771806}\cr
#'
#' @examples
#'
#' library(Matrix)
#' n <- 100
#' rank <- 2
#' clustertrue.y <- rep(1:rank, each = n/rank)
#' clustertrue.z <- rep(1:rank, each = n/rank)
#' A <- matrix(0, n, n)
#' for(i in 1:n) {
#'      for(j in 1:n) {
#'         A[i, j] <- ifelse(clustertrue.y[i] == clustertrue.z[i], rbinom(1, 1, 0.2), rbinom(1, 1, 0.1))
#'     }
#' }
#' diag(A) <- 0
#' A <- as(A, "dgCMatrix")
#' rsvd.pro(A, rank)
#'
#' @export rsvd.pro
#'
#'
rsvd.pro <- function(A, rank, p = 10, q = 2, dist = "normal", approA = FALSE, nthread = 1) {
    # Get coordinates of nonzero elements
    n <- nrow(A)
    Acoord <- sparse_matrix_coords(A, nthread)

    # The approximation for the column space
    # Set the reduced dimension for the column space
    ly <- round(rank) + round(p)

    # Generate a random test matrix Oy
    zsetseed(sample(2147483647L, 1))
    pre_alloc <- matrix(0, n, ly)
    Oy <- switch(dist,
                 normal = zrnormVec(pre_alloc),
                 unif = matrix(runif(ly*n), n, ly),
                 rademacher = matrix(sample(c(-1,1), (ly*n), replace = TRUE, prob = c(0.5,0.5)), n, ly),
                 stop("The sampling distribution is not supported!"))

    # Build the sketch matrix Y : Y = (A * A')^q * A * Oy
    Y <- spbin_power_prod(Acoord, Oy, q, nthread)

    # The approximation for the row space
    # Set the reduced dimension for the row space
    lz <- round(rank) + round(p)

    # Generate a random test matrix Oz
    Oz <- switch(dist,
                 normal = zrnormVec(pre_alloc),
                 unif = matrix(runif(lz*n), n, lz),
                 rademacher = matrix(sample(c(-1,1), (lz*n), replace = TRUE, prob = c(0.5,0.5)), n, lz),
                 stop("The sampling distribution is not supported!"))

    # Build sketch matrix Z : Z = (A' * A)^q * A' * Oz
    Z <- spbin_power_crossprod(Acoord, Oz, q, nthread)

    # Orthogonalize Y using QR decomposition: Y = Q * R1
    # Q <- qr_Q(Y)
    # Orthogonalize Z using QR decomposition: Z = T * R2
    # T <- qr_Q(Z)
    #
    # Compute two QR decompositions in parallel
    QT <- qr_Q2(Y, Z, nthread)
    Q <- QT[[1]]
    T <- QT[[2]]

    # Obtain the smaller matrix B := Q' * A * T
    B <- crossprod(spbin_power_crossprod(Acoord, Q, 0, nthread), T)

    # Compute the SVD of B and recover the approximated singular vectors of A
    fit <- svd(B)
    u <- Q %*% fit$u
    v <- T %*% fit$v
    d <- fit$d

    # Output the result
    if(approA == FALSE) {
        list(u = u, v = v, d = d)
    } else {
        # Compute the approximated matrix of the original A
        C  <- Q %*% B %*% t(T)
        list(u = u, v = v, d = d, approA = C)
    }
}
