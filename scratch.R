library(mvtnorm)

bekk.formula <- function(C, A, G, r, H.prev) {
    t(C) %*% C + t(A) %*% r %*% t(r) %*% A + t(G) %*% H.prev %*% G
}
