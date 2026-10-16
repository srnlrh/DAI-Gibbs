// =============================================================================
// Gibbs_Wrappers.cpp
// Single-iteration wrappers + master run_gibbs sampler
// Requires: RcppArmadillo, RcppZiggurat
// Both DAI_Gibbs.cpp and Gibbs_Corrected.cpp helpers (sample_trunc_normal,
// sample_trunc_normal_vec, std_multinorm) must be compiled in the same
// translation unit or sourced together via sourceCpp().
// =============================================================================

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include <Ziggurat.h>
// [[Rcpp::depends(RcppZiggurat)]]

#include <RcppArmadilloExtensions/sample.h>
#include <ctime>
#include <cmath>
#include <string>
#include <algorithm>

using namespace Rcpp;
using namespace arma;

// [[Rcpp::plugins("cpp11")]]

// ---- Forward declarations of helpers (defined in the companion .cpp files) --
// These are compiled in the same unit when you sourceCpp() all files together.
arma::vec std_multinorm(int p);
arma::vec sample_trunc_normal(const arma::vec& mean, const arma::mat& cov,
                              double lower, double upper);
arma::vec sample_trunc_normal_vec(const arma::vec& mean, const arma::mat& cov,
                                  const arma::vec& lower, const arma::vec& upper);
arma::umat iter_D_Mmat(arma::mat& Omg, arma::mat& nz_count, arma::mat& Omg_sum,
                           const arma::mat& S, const arma::mat& Ip,
                           int N, int p, double q, double r1, double s1,
                           bool post);
arma::umat iter_A_Mmat(arma::mat& Omg, arma::mat& nz_count, arma::mat& Omg_sum,
                           const arma::mat& S, const arma::mat& U,
                           const arma::mat& A, const arma::mat& Ip,
                           double k, int N, int p, double q, double r1, 
                           double s1, bool post);
arma::umat iter_D_DD(arma::mat& Omg, arma::mat& nz_count, arma::mat& Omg_sum,
                     arma::mat& alpha, arma::vec& beta, const arma::mat& S,
                     const arma::mat& Ip, int N, int p, double q, double r1, double s1,
                     bool post);
arma::umat iter_A_DD(arma::mat& Omg, arma::mat& nz_count, arma::mat& Omg_sum,
                     arma::mat& alpha, arma::vec& beta, const arma::mat& S,
                     const arma::mat& U, const arma::mat& A, const arma::mat& Ip,
                     double k, int N, int p, double q, double r1, double s1,
                     bool post);
  

  


// =============================================================================
// SHARED HELPER: finalise sparsity matrix and handle zero nz_count entries
// =============================================================================
static void finalise_sp(arma::umat& sp, arma::mat& nz_count, int nmc) {
  sp = arma::conv_to<arma::umat>::from(nz_count >= 0.5 * nmc);
  sp = arma::symmatu(sp);
  arma::uvec ind = arma::find(nz_count == 0);
  if (!ind.is_empty()) {
    nz_count.elem(ind).fill(1.0);
  }
}


// =============================================================================
// Master Gibbs sampler
//
// Arguments:
//   Y        — n x p data matrix
//   Omg_init — p x p initialisation
//   burnin   — number of burn-in iterations (discarded from summaries)
//   nmc      — number of post-burn-in samples to collect
//   q        — prior edge-inclusion probability
//   sampler  — "D", "A", or "I"
//   DD       — if true, use DD-constrained samplers from DAI_Gibbs [5];
//              otherwise use Gibbs_Corrected [6]
//   period   — for "I": D-step every `period`-th iteration, A-step otherwise
//   r1, s1   — hyperparameters for eta^2 prior
//
// Returns: List with samples (p x p x (burnin+nmc) cube),
//          nz_count (p x p), sparsity_matrix (p x p)
// =============================================================================
// [[Rcpp::export]]
List run_gibbs(const arma::mat& Y,
               const arma::mat& Omg_init,
               int burnin,
               int nmc,
               double q,
               std::string sampler = "D",
               bool DD             = false,
               int period          = 100,
               double r1           = 1e-4,
               double s1           = 1e-8) {
  
  // -- Input validation --
  if (sampler != "D" && sampler != "A" && sampler != "I")
    Rcpp::stop("'sampler' must be one of \"D\", \"A\", or \"I\".");
  
  int N = Y.n_rows;
  int p = Y.n_cols;
  int total_iter = burnin + nmc;
  
  arma::mat Ip = arma::eye<arma::mat>(p, p);
  arma::mat IN = arma::eye<arma::mat>(N, N);
  
  // -- Pre-compute persistent quantities --
  arma::mat S = Y.t() * Y;
  arma::mat Omg = Omg_init;
  arma::mat nz_count(p, p, arma::fill::zeros);
  arma::mat Omg_sum(p, p, arma::fill::zeros);
  arma::cube all_samples(p, p, total_iter);
  
  // Quantities needed for A-step and I-step
  double k = 0.0;
  arma::mat U, A_mat;
  if (sampler == "A" || sampler == "I") {
    k     = (arma::eig_sym(S)).back() * 1.001;
    U     = Y.t() / std::sqrt(k);
    if (N < p) {
      arma::mat tmp = IN - U.t() * U;
      tmp   = (tmp + tmp.t()) / 2.0;
      A_mat = arma::chol(tmp, "lower");
    } else {
      A_mat = k * Ip - S;
      A_mat = (A_mat + A_mat.t()) / 2.0;
    }
  }
  
  // DD bookkeeping — only needed for DD = true
  arma::mat alpha(p, p, arma::fill::zeros);
  arma::vec beta(p, arma::fill::zeros);
  if (DD) {
    alpha = arma::repmat(arma::sum(Omg, 1), 1, p) - Omg;
    beta  = arma::sum(arma::abs(Omg), 1) - diagvec(Omg);
  }
  
  clock_t time_start = clock();
  
  // -- Main loop --
  for (int iter = 0; iter < total_iter; ++iter) {
    
    bool post = (iter >= burnin);
    
    if (sampler == "D") {
      if (DD) {
        iter_D_DD(Omg, nz_count, Omg_sum, alpha, beta,
                  S, Ip, N, p, q, r1, s1, post);
      } else {
        iter_D_Mmat(Omg, nz_count, Omg_sum,
                        S, Ip, N, p, q, r1, s1, post);
      }
      
    } else if (sampler == "A") {
      if (DD) {
        iter_A_DD(Omg, nz_count, Omg_sum, alpha, beta,
                  S, U, A_mat, Ip, k, N, p, q, r1, s1, post);
      } else {
        iter_A_Mmat(Omg, nz_count, Omg_sum,
                        S, U, A_mat, Ip, k, N, p, q, r1, s1, post);
      }
      
    } else {
      // "I": D-step at iterations 0, period, 2*period, ... ; A-step otherwise
      // Matches 0-indexed C++ convention iter % period == 0 from both files [5][6]
      if (iter % period == 0) {
        if (DD) {
          iter_D_DD(Omg, nz_count, Omg_sum, alpha, beta,
                    S, Ip, N, p, q, r1, s1, post);
        } else {
          iter_D_Mmat(Omg, nz_count, Omg_sum,
                          S, Ip, N, p, q, r1, s1, post);
        }
      } else {
        if (DD) {
          iter_A_DD(Omg, nz_count, Omg_sum, alpha, beta,
                    S, U, A_mat, Ip, k, N, p, q, r1, s1, post);
        } else {
          iter_A_Mmat(Omg, nz_count, Omg_sum,
                          S, U, A_mat, Ip, k, N, p, q, r1, s1, post);
        }
      }
    }
    
    all_samples.slice(iter) = Omg;
  }
  
  // -- Finalise sparsity matrix --
  arma::umat sp;
  finalise_sp(sp, nz_count, nmc);
  
  clock_t time_end = clock();
  double time_taken = (time_end - time_start) / (double)CLOCKS_PER_SEC;
  
  return List::create(
    Named("samples")         = all_samples,
    Named("nz_count")        = nz_count,
    Named("sparsity_matrix") = sp,
    Named("Time_Taken")      = time_taken
  );
}