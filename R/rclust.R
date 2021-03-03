
#' Randomized spectral clustering using random sampling or random projection
#'
#' Randomized spectral clustering for undirected networks. The clusters are computed using two random schemes, namely, the random sampling
#' and the random projection scheme. Can deal with very large networks.
#'
#' This function computes the clusters of undirected networks using
#' randomized spectral clustering algorithms. The random projection-based eigendecomposition or the
#' random sampling-based eigendecomposition is first computed for the adjacency matrix of the undirected network.
#' The k-means is then performed on the randomized eigen vectors.
#'
#'
#'
#' @param A The adjacency matrix of an undirected network with type "dgCMatrix".
#' @param method The method for computing the randomized eigendecomposition. Random sampling-based eigendecomposition
#'               is implemented if \code{method="rsample"}, and random projection-based
#'               eigendecomposition is implemented if \code{method="rproject"}.
#' @param k The number of target clusters.
#' @param p The oversampling parameter in the random projection scheme. Requested only
#'          if \code{method="rproject"}. Default is 10.
#' @param q The power parameter in the random projection scheme. Requested only if
#'          \code{method="rproject"}. Default is 2.
#' @param dist The distribution of the entry of the random test matrix in the random
#'             projection scheme. Requested only if \code{method="rproject"}. Default
#'             is \code{"normal"}.
#' @param P The sampling probability in the random sampling scheme. Requested only
#'          if \code{method="rsample"}.
#' @param iter.max Maximum number of iterations in the \code{\link[stats]{kmeans}}.
#'                 Default is 50.
#' @param nstart The number of random sets in \code{\link[stats]{kmeans}}. Default is 10.
#' @param ... Additional arguments.
#'
#' @return \item{cluster}{The cluster vector (from \code{1:k}) with the numbers indicating which
#'              cluster each node is assigned.}
#'         \item{rvectors}{The randomized \code{k} eigen vectors computed by
#'              \code{\link[RandClust]{reig.pro}} or \code{\link[RandClust]{reig.sam}}.}
#' @export rclust
#' @seealso \code{\link[RandClust]{reig.pro}}, \code{\link[RandClust]{reig.sam}}.
#' @examples
#' n <- 100
#' k <- 2
#' clustertrue <- rep(1:k, each = n/k)
#' A <- matrix(0, n, n)
#' for(i in 1: (n-1)) {
#'    for(j in (i+1):n) {
#'        A[i, j] <- ifelse(clustertrue[i] == clustertrue[j], rbinom(1, 1, 0.2), rbinom(1, 1, 0.1))
#'        A[j, i] <- A[i, j]
#'     }
#' }
#' A <- as(A, "dgCMatrix")
#' rclust(A, method = "rsample", k = k, P = 0.7)
#'
#'
rclust <- function(A, method = c("rsample", "rproject"), k, p = 10, q = 2, dist = "normal", P, iter.max = 50, nstart = 10, ...) {

  #Compute the randomized eigen vectors
  if(method == "rsample"){
    sameig <- reig.sam (A = A, P = P, k = k, ...)
    A.u <- sameig$vectors
  }

  if(method == "rproject"){
    projeig <- reig.pro (A = A, rank = k, p = p, q = q, dist = dist, ...)
    A.u <- projeig$vectors[, 1 : k]
  }




  #Find the clusters using the k-means

  cluster <- rep(0, nrow(A))
  fit <- kmeans(A.u, k, iter.max = iter.max, nstart = nstart)
  cluster <- fit$cluster


  list(cluster = cluster, rvectors = A.u)
}

