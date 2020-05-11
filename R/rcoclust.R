
#' Randomized spectral co-clustering using random sampling or random projection
#'
#' Randomized spectral co-clustering for directed networks. The row clusters and
#' column clusters are computed using two random schemes, namely, the random sampling
#' and the random projection scheme. Can deal with very large networks.
#'
#' This function computes the row clusters and column clusters of directed networks using
#' randomized spectral co-clustering algorithms. The random projection-based SVD or the
#' random sampling-based SVD is first computed for the adjacency matrix of the directed network.
#' The k-means or the spherical k-median is then performed on the randomized singular vectors.
#'
#'
#'
#' @param A The adjacency matrix of a directed network with type "dgCMatrix".
#' @param method The method for computing the randomized SVD. Random sampling-based SVD
#'               is implemented if \code{method="rsample"}, and random projection-based
#'               SVD is implemented if \code{method="rproject"}.
#' @param ky The number of target row clusters.
#' @param kz The number of target column clusters.
#' @param p The oversampling parameter in the random projection scheme. Requested only
#'          if \code{method="rproject"}. Default is 10.
#' @param q The power parameter in the random projection scheme. Requested only if
#'          \code{method="rproject"}. Default is 2.
#' @param dist The distribution of the entry of the random test matrix in the random
#'             projection scheme. Requested only if \code{method="rproject"}. Default
#'             is \code{"normal"}.
#' @param P The sampling probability in the random sampling scheme. Requested only
#'          if \code{method="rsample"}.
#' @param normalize The k-means is implemented if \code{FALSE}. Otherwise the spherical
#'                  k-median is implemented. Default is \code{FALSE}.
#' @param iter.max Maximum number of iterations in the \code{\link[stats]{kmeans}} and
#'                 that for choosing the starting point of \code{\link[Gmedian]{kGmedian}}.
#'                 Default is 50.
#' @param nstartkmedian The number of times the k-median algorithm is ran in the
#'                     \code{\link[Gmedian]{kGmedian}}. Default is 10.
#' @param nstartkmeans The number of random sets in \code{\link[stats]{kmeans}} and that
#'                     for choosing the starting point of \code{\link[Gmedian]{kGmedian}}. Default is 10.
#' @param ... Additional arguments.
#'
#' @return \item{cluster.y}{The row cluster vector (from \code{1:ky}) with the numbers indicating which
#'              row cluster each node is assigned.}
#'         \item{cluster.z}{The column cluster vector (from \code{1:kz}) with the numbers indicating which
#'              column cluster each node is assigned.}
#'         \item{rvectors.y}{The randomized left \code{min(ky,kz)} singular vectors computed by
#'              \code{\link[RandClust]{rsvd.pro}} or \code{\link[RandClust]{rsvd.sam}}.}
#'         \item{rvectors.z}{The randomized right \code{min(ky,kz)} singular vectors computed by
#'              \code{\link[RandClust]{rsvd.pro}} or \code{\link[RandClust]{rsvd.sam}}.}
#' @export rcoclust
#' @seealso \code{\link[RandClust]{rsvd.pro}}, \code{\link[RandClust]{rsvd.sam}}.
#' @examples
#' n <- 120
#' ky <- 2
#' kz <- 3
#' cluster.y <- rep(1:ky, each = n/ky)
#' cluster.z <- rep(1:kz, each = n/kz)
#' probmat <- matrix(0.2, ky, kz)
#' diag(probmat) <- 0.1
#' A <- sample_scbm(type = "scbm", cluster.y, cluster.z, probmat = probmat, graph = FALSE)
#' rcoclust(A, method ="rsample", ky, kz, P = 0.7, normalize = FALSE)
#' rcoclust(A, method ="rproject", ky, kz, normalize = TRUE)
#'
#'
rcoclust <- function(A, method = c("rsample", "rproject"), ky, kz, p = 10, q = 2, dist = "normal", P, normalize = FALSE, iter.max = 50, nstartkmedian = 10, nstartkmeans = 10, ...){

  #Compute the randomized singular vectors
  if(method == "rsample"){
    samsvd <- rsvd.sam (A = A, P = P, nu = min(ky, kz), nv = min(ky, kz), ...)
    A.u <- samsvd$u
    A.v <- samsvd$v
  }

  if(method == "rproject"){
    projsvd <- rsvd.pro (A = A, rank = min(ky, kz), p = p, q = q, dist = dist, ...)
    A.u <- projsvd$u[, 1 : min(ky, kz)]
    A.v <- projsvd$v[, 1 : min(ky, kz)]
  }



	#Find the clusters using the k-means if 'normalize==FALSE' and using the spherical k-median if 'normalize==TRUE'
  cluster.y <- rep(0, nrow(A))
  cluster.z <- rep(0, nrow(A))

	if(normalize == FALSE){
	    fit.y <- kmeans(A.u, ky, iter.max = iter.max, nstart = nstartkmeans)
	    fit.z <- kmeans(A.v, kz, iter.max = iter.max, nstart = nstartkmeans)
	    cluster.y <- fit.y$cluster
	    cluster.z <- fit.z$cluster
	}


	if(normalize == TRUE){
	    norm.u <- apply(A.u, 1, crossprod)
	    norm.v <- apply(A.v, 1, crossprod)
      zero.u <- which(norm.u == 0)
	    zero.v <- which(norm.v == 0)
	    if(length(zero.u) != 0){
	        A.un <- diag(1/sqrt(norm.u[-zero.u])) %*% A.u[-zero.u, ]
	        fit.y <- kGmedian(A.un, ncenters = ky, nstart = nstartkmedian, nstartkmeans = nstartkmeans, iter.max = iter.max)
	        cluster.y[-zero.u] <- fit.y$cluster
			    cluster.y[zero.u] <- 1
	    }

	    if(length(zero.u)==0){
	        A.un<- diag(1/sqrt(norm.u))%*%A.u
	        fit.y <- kGmedian(A.un, ncenters = ky, nstart = nstartkmedian, nstartkmeans = nstartkmeans, iter.max = iter.max)
	        cluster.y <- fit.y$cluster
	    }

	    if(length(zero.v) != 0){
	        A.vn <- diag(1/sqrt(norm.v[-zero.v]))%*%A.v[-zero.v, ]
	        fit.z <- kGmedian(A.vn, ncenters = kz, nstart = nstartkmedian, nstartkmeans = nstartkmeans, iter.max = iter.max)
	        cluster.z[-zero.v] <- fit.z$cluster
			    cluster.z[zero.v] <- 1
	    }

	    if(length(zero.v) == 0){
	        A.vn <- diag(1/sqrt(norm.v))%*%A.v
	        fit.z <- kGmedian(A.vn, ncenters = kz, nstart = nstartkmedian, nstartkmeans = nstartkmeans, iter.max = iter.max)
	        cluster.z <- fit.z$cluster
        }

	}

	list(cluster.y = cluster.y, cluster.z = cluster.z, rvectors.y = A.u, rvectors.z = A.v)

}

