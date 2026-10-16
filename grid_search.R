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
# grid_search
#
# Performs a grid search over candidate latent dimensions (qvals) to select
# the value of q that maximises the Gaussian log-likelihood on the test set.
#
# Arguments:
#   test    - numeric matrix of held-out (test) observations  [n_test  x p]
#   train   - numeric matrix of training observations         [n_train x p]
#   qvals   - integer vector of candidate latent dimensions to evaluate
#   sampler - sampler type passed to estimate_precision (default: "I")
#   DD      - logical; if FALSE, a small diagonal ridge is added to ensure
#             positive definiteness when eigenvalues are near zero (default: FALSE)
#
# Returns:
#   The value in qvals that maximises the Gaussian log-likelihood on the test set.
# ---------------------------------------------------------------------------
grid_search <- function(test, train, qvals, sampler = "I", DD = FALSE) {
  
  # --- Input validation -----------------------------------------------------
  if (!is.matrix(train) || !is.numeric(train))
    stop("`train` must be a numeric matrix.")
  if (!is.matrix(test) || !is.numeric(test))
    stop("`test` must be a numeric matrix.")
  if (ncol(train) != ncol(test))
    stop("`train` and `test` must have the same number of columns (variables).")
  if (!is.numeric(qvals) || length(qvals) == 0 || any(qvals < 1))
    stop("`qvals` must be a non-empty vector of positive integers.")
  if (!is.logical(DD))
    stop("`DD` must be logical (TRUE or FALSE).")
  
  p       <- ncol(train)
  n_train <- nrow(train)
  n_test  <- nrow(test)
  
  # Pre-allocate log-likelihood storage (one entry per candidate q)
  l <- numeric(length(qvals))
  
  # Sample covariance of the test set — used in the trace term of the
  # Gaussian log-likelihood: log p(X_test | Omega) ∝
  #   (n_test/2) * log|Omega| - (n_test/2) * tr(Omega %*% S_test)
  S <- var(test)
  if (min(eigen(S)$values) < 1e-10) {
    rd <- max(1e-10, -min(eigen(S)$values)) + 0.01
    S <- S + diag(p) * rd
  }
  
  # --- Grid search ----------------------------------------------------------
  for (k in seq_along(qvals)) {
    
    # Estimate the precision matrix via the Gibbs sampler for this q
    estimated_precision <- tryCatch(
      estimate_precision(X = train, q = qvals[k],
                         Omega_init = diag(p), sampler = sampler),
      error = function(e) {
        warning(sprintf("estimate_precision failed for q = %d: %s", qvals[k], e$message))
        return(NULL)
      }
    )
    
    # Skip this q if estimation failed
    if (is.null(estimated_precision)) {
      l[k] <- -Inf
      next
    }
    
    # If not using a positive-definite-preserving sampler (DD = FALSE),
    # check for near-zero or negative eigenvalues and apply a ridge correction
    if (!DD) {
      min_eig <- min(eigen(estimated_precision, symmetric = TRUE)$values)
      if (min_eig < 1e-10) {
        ridge <- max(1e-10, -min_eig) + 0.01
        estimated_precision <- estimated_precision + diag(p) * ridge
        warning(sprintf("Precision matrix for q = %d was not positive definite; ",
                        "ridge correction of %.4f applied.", qvals[k], ridge))
      }
    }
    
    # Gaussian log-likelihood on the test set (up to a constant):
    #   (n_test/2) * log|Omega| - (n_test/2) * tr(Omega * S)
    log_det_term  <- (n_test / 2) * log(det(estimated_precision))
    trace_term    <- (n_test / 2) * sum(estimated_precision * S)
    l[k]          <- log_det_term - trace_term
  }
  
  # Return the q value with the highest log-likelihood
  return(qvals[which.max(l)])
}