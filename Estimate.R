library(Rcpp)
library(RcppArmadillo)
library(RcppZiggurat)
library(RcppTN)
library(MASS)
library(Matrix)

sourceCpp(file = 'seed.cpp')
sourceCpp(file = 'sample_multinormal.cpp')
sourceCpp(file = 'sample_truncated_normal.cpp')
sourceCpp(file = 'DGIBBS_one_iter.cpp')
sourceCpp(file = 'AGIBBS_one_iter.cpp')
sourceCpp(file = 'Gibbs_sampler.cpp')

# Set seeds for reproducibility across R, Armadillo, and Ziggurat RNGs
set.seed(1234)
set_seed_arma(1234)
set_seed_zigg(1234)


# ---------------------------------------------------------------------------
# estimate_precision
#
# Estimates a sparse precision matrix Omega using a Gibbs sampler with
# optional low-rank structure of dimension q. Posterior samples are averaged
# over the retained (post-burnin) MCMC draws, masked by the inferred
# sparsity pattern.
#
# Arguments:
#   X          - numeric data matrix                              [n x p]
#   q          - integer; latent rank / number of factors
#   Omega_init - numeric p x p matrix; initial value for Omega
#                (must be symmetric positive definite)
#   burnin     - integer; number of burn-in iterations to discard (default: 2000)
#   nmc        - integer; number of post-burn-in samples to retain (default: 1000)
#   sampler    - character; sampler type, e.g. "I"               (default: "I")
#   DD         - logical; positive-definite sampler flag          (default: FALSE)
#   period     - integer; thinning period                         (default: 100)
#   r1, s1     - numeric; hyperparameters for the prior           (defaults: 1e-4, 1e-8)
#
# Returns:
#   A p x p sparse precision matrix estimate (posterior mean masked by
#   the inferred sparsity pattern).
# ---------------------------------------------------------------------------
estimate_precision <- function(X, q, Omega_init, burnin = 2000, nmc = 1000,
                               sampler = "I", DD = FALSE, period = 100,
                               r1 = 1e-4, s1 = 1e-8) {
  
  # --- Input validation -----------------------------------------------------
  if (!is.matrix(X) || !is.numeric(X))
    stop("`X` must be a numeric matrix.")
  if (!is.numeric(q) || length(q) != 1 || q < 1 || q != floor(q))
    stop("`q` must be a single positive integer.")
  if (!is.matrix(Omega_init) || !isSymmetric(Omega_init))
    stop("`Omega_init` must be a symmetric matrix.")
  
  p <- ncol(X)
  n <- nrow(X)
  
  if (nrow(Omega_init) != p || ncol(Omega_init) != p)
    stop("`Omega_init` must be a p x p matrix where p = ncol(X).")
  if (min(eigen(Omega_init, symmetric = TRUE)$values) <= 0)
    stop("`Omega_init` must be positive definite.")
  if (burnin < 0 || nmc < 1)
    stop("`burnin` must be non-negative and `nmc` must be at least 1.")
  
  # --- Run Gibbs sampler ----------------------------------------------------
  # Pass all control parameters through to the C++ sampler
  obj <- run_gibbs(
    X          = X,
    Omega_init = Omega_init,   # use the caller-supplied initial value
    q          = q,
    burnin     = burnin,
    nmc        = nmc,
    sampler    = sampler,
    DD         = DD,
    period     = period,
    r1         = r1,
    s1         = s1
  )
  
  # --- Extract outputs ------------------------------------------------------
  sp <- obj$sparsity_matrix   # p x p binary sparsity mask
  nz <- obj$nz_count          # p x p matrix indicating number of non-zero 
                              # samples corresponding to each entry
  
  # Sanity check: sparsity matrix must be p x p
  if (!is.matrix(sp) || any(dim(sp) != c(p, p)))
    stop("Gibbs sampler returned a malformed sparsity matrix.")
  
  # Sanity check: non-zero count matrix must be p x p
  if (!is.matrix(nz) || any(dim(nz) != c(p, p)))
    stop("Gibbs sampler returned a malformed matrix of non-zero counts.")
  
  # Retain only post-burnin samples: array of dimension [p x p x nmc]
  post_samples <- obj$samples[, , (burnin + 1):(burnin + nmc)]
  
  # Check the sample array has the expected dimensions
  if (length(dim(post_samples)) != 3 || dim(post_samples)[3] != nmc)
    stop("Post-burnin sample array has unexpected dimensions.")
  
  # Compute posterior mean across non-zero retained MCMC samples (3rd dimension)
  # then zero out entries outside the inferred sparsity pattern
  Omega_hat <- apply(post_samples, c(1, 2), sum) * sp / nz
  
  # Replace with the following if aim is to average over all retained samples
  # Here, instead of 'nz', the denominator is simply nmc
  # Omega_hat <- apply(post_samples, c(1, 2), mean) * sp
  
  return(Omega_hat)
}