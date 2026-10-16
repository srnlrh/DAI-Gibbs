# =============================================================================
# compute_metrics.R
#
# Computes the four evaluation metrics reported in the paper:
#   1. MCC    — Matthews Correlation Coefficient on the sparsity pattern
#   2. F-score — harmonic mean of precision and recall on the sparsity pattern
#   3. Relative Estimation Error — ||Omega_hat - Omega_true||_norm / ||Omega_true||_norm
#   4. Proportion of PD samples — fraction of MCMC draws with min eigenvalue > 1e-15
#
# Arguments:
#   Omega_true   - p x p true precision matrix
#   Omega_hat    - p x p estimated precision matrix (posterior mean or point estimate)
#   norm_type    - character; norm type passed to base::norm(). One of:
#                  "F" (Frobenius, default; used in the paper), "1", "I", "2", "M"
#   samples      - optional p x p x T array of post-burnin MCMC draws.
#                  Required only for computing proportion of PD samples.
#   pd_threshold - minimum eigenvalue threshold for declaring a sample PD
#                  (default: 1e-15, as specified)
#
# Returns:
#   A named list with elements: MCC, F_score, rel_error, prop_PD
#   (prop_PD is NA if samples is not provided)
# =============================================================================

compute_metrics <- function(Omega_true,
                            Omega_hat,
                            norm_type    = "F",
                            samples      = NULL,
                            pd_threshold = 1e-15) {
  
  # --- Input validation -------------------------------------------------------
  if (!is.matrix(Omega_true) || !is.matrix(Omega_hat))
    stop("`Omega_true` and `Omega_hat` must both be matrices.")
  if (!identical(dim(Omega_true), dim(Omega_hat)))
    stop("`Omega_true` and `Omega_hat` must have the same dimensions.")
  if (!norm_type %in% c("F", "1", "I", "2", "M"))
    stop("`norm_type` must be one of: \"F\", \"1\", \"I\", \"2\", \"M\".")
  
  p <- nrow(Omega_true)
  
  # --- Helper: extract upper-triangle indices (off-diagonal entries only) -----
  # Graph structure is determined solely by off-diagonal sparsity pattern
  upper_idx <- which(upper.tri(Omega_true))
  
  # True and estimated binary sparsity patterns (1 = edge present, 0 = absent)
  true_pattern <- as.integer(Omega_true[upper_idx] != 0)
  est_pattern  <- as.integer(Omega_hat[upper_idx]  != 0)
  
  # --- Metric 1 & 2: MCC and F-score ------------------------------------------
  # Confusion matrix entries
  TP <- sum(est_pattern == 1 & true_pattern == 1)
  FP <- sum(est_pattern == 1 & true_pattern == 0)
  TN <- sum(est_pattern == 0 & true_pattern == 0)
  FN <- sum(est_pattern == 0 & true_pattern == 1)
  
  # MCC: Matthews Correlation Coefficient
  # Numerator and denominator computed separately to handle zero-division cleanly
  mcc_denom <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  MCC <- if (mcc_denom == 0) 0 else (TP * TN - FP * FN) / mcc_denom
  
  # F-score: harmonic mean of precision and recall
  precision <- if ((TP + FP) == 0) 0 else TP / (TP + FP)
  recall    <- if ((TP + FN) == 0) 0 else TP / (TP + FN)
  F_score   <- if ((precision + recall) == 0) 0 else
    2 * precision * recall / (precision + recall)
  
  # --- Metric 3: Relative Estimation Error ------------------------------------
  # Scaled by the norm of the true matrix, as reported in the paper (Table 3)
  # Default norm is Frobenius ("F"), matching the paper exactly
  norm_error <- norm(Omega_hat - Omega_true, type = norm_type)
  norm_true  <- norm(Omega_true,             type = norm_type)
  rel_error  <- if (norm_true == 0) NA else norm_error / norm_true
  
  # --- Metric 4: Proportion of PD samples ------------------------------------
  # Requires the full MCMC sample array [p x p x T]
  prop_PD <- NA
  if (!is.null(samples)) {
    if (length(dim(samples)) != 3 || dim(samples)[1] != p || dim(samples)[2] != p)
      stop("`samples` must be a p x p x T array with the same p as `Omega_true`.")
    
    T_samp <- dim(samples)[3]
    
    # A draw is PD if its smallest eigenvalue exceeds pd_threshold
    is_pd <- vapply(seq_len(T_samp), function(t) {
      min(eigen(samples[, , t], symmetric = TRUE, only.values = TRUE)$values) > pd_threshold
    }, logical(1))
    
    prop_PD <- mean(is_pd)
  }
  
  # --- Return -----------------------------------------------------------------
  return(list(
    MCC       = MCC,
    F_score   = F_score,
    rel_error = rel_error,  # computed using norm_type (default: Frobenius)
    prop_PD   = prop_PD     # NA if samples not provided
  ))
}