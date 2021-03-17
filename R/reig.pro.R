#' Compute randomized eigenvalue decomposition of a symmetric matrix using random projection
#'
#' Compute the randomized eigenvalue decomposition of a symmetric matrix by random projection.
#' The randomized eigen vectors and eigen values are computed. Can deal with very large
#' data matrix.
#'
#' This function computes the randomized eigen value decomposition of a data matrix using the random
#' projection scheme. The data matrix \code{A} is symmetric. It is first compressed to a
#' smaller matrix with its columns (rows) being the linear combinations of the
#' columns (rows) of \code{A}. The classical eigen value decomposition is then performed on the smaller
#' matrix. The randomized eigen value decomposition of \code{A} are obtained by postprocessing.
#'
#'
#'
#'
#' @param A Input data matrix of class "\code{dgCMatrix}". The matrix should be a symmetric matrix but not necessarily be the adjacency matrix of a network.
#' @param rank The target rank of the low-rank decomposition.
#' @param p The oversampling parameter. It need to be a positive integer number. Default value is 10.
#' @param q The power parameter. It need to be a positive integer number. Default value is 2.
#' @param dist The distribution of the entry of the random test matrix. Can be \code{"normal"} (standard normal distribution),
#'             \code{"unif"} (uniform distribution from -1 to 1), or \code{"rademacher"} (randemacher distribution). Default
#'             is \code{"normal"}.
#' @param approA A logical variable indicating whether the approximated \code{A} is returned. Default is \code{FALSE}.
#'
#' @return \item{vectors}{The randomized \code{rank+p} eigen vectors.}
#'         \item{values}{The \code{rank+p} eigen values.}
#'         \item{approA}{The approximated data matrix if requested.}
#'
#' @references N. Halko, P.-G. Martinsson, and J. A. Tropp. (2011)
#' \emph{Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions},
#' \emph{SIAM review, Vol. 53(2), 217-288}\cr
#' \url{https://epubs.siam.org/doi/10.1137/090771806}\cr
#'
#' @examples
#'
#' n <- 100
#' rank <- 2
#' clustertrue <- rep(1:rank, each = n/rank)
#' A <- matrix(0, n, n)
#' for(i in 1:(n - 1)) {
#'     for(j in (i + 1):n) {
#'         A[i, j] <- ifelse(clustertrue[i] == clustertrue[j], rbinom(1, 1, 0.2), rbinom(1, 1, 0.1))
#'     }
#' }
#' diag(A) <- 0
#' A <- A + t(A)
#' A <- as(A, "dgCMatrix")
#' reig.pro(A, rank)
#'
#' @export reig.pro
#'
#'
reig.pro <- function(A, rank, p = 10, q = 2, dist = "normal", approA = FALSE) {
    # Dim of input matrix
    n <- nrow(A)

    # Get coordinates of nonzero elements
    Acoord <- sparse_matrix_coords(A)

	  # Set the reduced dimension
    l <- round(rank) + round(p)

    # Generate a random test matrix O
    zsetseed(sample(2147483647L, 1))
    pre_alloc <- matrix(0, n, l)
    O <- switch(dist,
                normal = zrnormVec(pre_alloc),
                unif = matrix(runif(l*n), n, l),
                rademacher = matrix(sample(c(-1,1), (l*n), replace = TRUE, prob = c(0.5,0.5)), n, l),
                stop("The sampling distribution is not supported!"))

    # Build sketch matrix Y : Y = A * O
	  Y <- spbin_power_prod(Acoord, O, q = 0)

    # Orthogonalize Y using QR decomposition: Y=QR
    if( q > 0 ) {
        for( i in 1:q ) {
            Y <- qr_Q(Y)
            Z <- spbin_power_crossprod(Acoord, Y, q = 0)
            Z <- qr_Q(Z)
            Y <- spbin_power_prod(Acoord, Z, q = 0)
        }
    }
    Q <- qr_Q(Y)

    # Obtain the smaller matrix B := Q' * A * Q
    B <- crossprod(spbin_power_crossprod(Acoord, Q, q = 0), Q)

	  # Compute the eigenvalue decomposition of B and recover the approximated eigenvectors of A
	  fit <- eigen(B, symmetric = TRUE)
	  o <- order(abs(fit$values),decreasing=T)
	  vectors <- Q %*% fit$vectors[,o]
	  values <- fit$values[o]

	  # Output the result
	  if(approA == FALSE) {
	      list(vectors = vectors, values =values)
	  } else {
	      # Compute the approximated matrix of the original A
	      C  <- Q %*% B %*% t(Q)
	      list(vectors = vectors, values = values, approA = C)
	  }
}
