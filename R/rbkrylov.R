rbkrylov <- function(A, rank, q = 2, dist = "normal", nthread = 1, tol = 1e-5, ...) {
    # Get coordinates of nonzero elements
    n <- nrow(A)
    Acoord <- sparse_matrix_coords(A, nthread)

    # Generate a random test matrix Pi
    zsetseed(sample(2147483647L, 1))
    pre_alloc <- matrix(0, n, rank)
    Pi <- switch(dist,
                 normal = zrnormVec(pre_alloc),
                 unif = matrix(runif(rank * n), n, rank),
                 rademacher = matrix(sample(c(-1, 1), rank * n, replace = TRUE, prob = c(0.5, 0.5)), n, rank),
                 stop("The sampling distribution is not supported!"))

    # Build the Krylov space:
    # K = [A * Pi, (AA') * A * Pi, ..., (AA')^q * A * Pi]
    K <- spbin_krylov_space(Acoord, Pi, q, nthread)
    rm(Pi)
    gc()

    # Orthogonalize K using QR decomposition: K = Q * R
    qr_Q_inplace(K)
    Q <- K

    # Obtain the smaller matrix B := A' * Q
    B <- spbin_power_crossprod(Acoord, Q, 0, nthread)

    # Compute the partial SVD of B and recover the approximated singular vectors of A
    #partialsvd <- irlba(B, nu = 0, nv = rank, tol = tol, ...)
    #Z <- Q %*% partialsvd$v

    partialsvd <- svd(B)
    Z <- Q %*% partialsvd$v[, 1:rank]
    # Output the result
    Z
}
