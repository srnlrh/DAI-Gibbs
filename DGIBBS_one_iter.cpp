// =============================================================================
// DGIBBS_one_iter.cpp
// Single-iteration wrappers for D-Gibbs
// Requires: RcppArmadillo, RcppZiggurat
// All helpers (sample_trunc_normal, sample_trunc_normal_vec, std_multinorm) 
// must be compiled in the same translation unit or sourced together via 
// sourceCpp().
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


// =============================================================================
// FUNCTION 1: one iteration of D-Gibbs (For M-matrix)  
//
// Arguments (all by reference, mutated in place):
//   Omg      — current p x p sample (updated)
//   nz_count — running non-zero counter (updated if post-burnin)
//   Omg_sum  — running sum (updated if post-burnin)
//   S        — scatter matrix Y'Y (precomputed)
//   Ip       — p x p identity (precomputed)
//   N, p     — dimensions
//   q, r1, s1 — hyperparameters
//   post     — true if this is a post-burnin iteration
// Returns: current-iteration sparsity matrix (off-diagonal 0/1)
// =============================================================================
static arma::umat iter_D_Mmat(arma::mat& Omg,
                                  arma::mat& nz_count,
                                  arma::mat& Omg_sum,
                                  const arma::mat& S,
                                  const arma::mat& Ip,
                                  int N, int p,
                                  double q, double r1, double s1,
                                  bool post) {
  
  // -- Sample global shrinkage --
  double eta_sq_gam = randg(distr_param(
    r1 + 0.5 * (arma::accu(Omg != 0) - p) / 2.0,
    1.0 / (s1 + (arma::accu(Omg % Omg) - arma::sum(Omg.diag() % Omg.diag())) / 2.0)));
  double tau_sq = 1.0 / eta_sq_gam;
  
  double bin;
  arma::vec mean_v(1), var_v(1);
  
  for (int i = 0; i < p; ++i) {
    
    // -- Off-diagonal entries j > i --
    for (int j = i + 1; j < p; ++j) {
      double aij = S(i,i)/N + S(j,j)/N + 1.0/(N * tau_sq);
      double bij = -arma::sum(Omg.col(i) % S.col(j) / N +
                              Omg.col(j) % S.col(i) / N) +
                              Omg(i,j) * (S(i,i)/N + S(j,j)/N);
      double cij = 2.0 * q / (1.0 - q) *
        std::sqrt(1.0 / (N * aij * tau_sq)) *
        std::exp(N * bij * bij / (2.0 * aij)) *
        arma::normcdf(bij / std::sqrt((double)N / aij));
      
      if (std::isinf(cij)) {
        bin = 1.0;
      } else {
        double q_star = cij / (1.0 + cij);
        bin = R::rbinom(1, q_star);
      }
      
      if (bin == 1.0) {
        arma::vec a(1);  a(0) = aij;
        Omg(i,j) = Omg(j,i) = (sample_trunc_normal(bij / a, 1.0 / (N * a),
                       R_NegInf, 0.0))(0);
      } else {
        Omg(i,j) = Omg(j,i) = 0.0;
      }
      if (post) {
        nz_count(i,j) = nz_count(j,i) = nz_count(i,j) + bin;
      }
    }
    
    // -- Diagonal entry --
    double bd = arma::sum(Omg.col(i) % S.col(i) / N) - Omg(i,i) * S(i,i) / N;
    if (N > p) {
      mean_v(0) = N * (1.0 - bd) / S(i,i);
      var_v(0)  = 1.0 / S(i,i);
    } else {
      mean_v(0) = N * (1.0 - bd) / (S(i,i) + 0.01);
      var_v(0)  = 1.0 / (S(i,i) + 0.01);
    }
    Omg(i,i) = sample_trunc_normal(mean_v, var_v, 0.0, arma::datum::inf)(0);
  }
  
  if (post) {
    Omg_sum  += Omg;
    nz_count += Ip;
  }
  
  arma::umat sp_iter = arma::conv_to<arma::umat>::from(Omg != 0);
  sp_iter.diag().zeros();
  return sp_iter;
}




// =============================================================================
// FUNCTION 2: one iteration of D-Gibbs  (DAI_Gibbs — DD constraint)
// Extra state: alpha (p x p), beta (p-vec) — mutated in place
// =============================================================================
static arma::umat iter_D_DD(arma::mat& Omg,
                            arma::mat& nz_count,
                            arma::mat& Omg_sum,
                            arma::mat& alpha,
                            arma::vec& beta,
                            const arma::mat& S,
                            const arma::mat& Ip,
                            int N, int p,
                            double q, double r1, double s1,
                            bool post) {
  
  // -- Sample global shrinkage --
  double eta_sq_gam = randg(distr_param(
    r1 + 0.5 * (arma::accu(Omg != 0) - p) / 2.0,
    1.0 / (s1 + (arma::accu(Omg % Omg) - arma::sum(Omg.diag() % Omg.diag())) / 2.0)));
  double tau_sq = 1.0 / eta_sq_gam;
  
  double bin;
  arma::vec mean_v(1), var_v(1);
  
  for (int i = 0; i < p; ++i) {
    
    // -- Diagonal: lower bound beta[i] (DD constraint) --
    double bd = arma::sum(Omg.col(i) % S.col(i) / N) - Omg(i,i) * S(i,i) / N;
    mean_v(0) = N * (1.0 - bd) / S(i,i);
    var_v(0)  = 1.0 / S(i,i);
    Omg(i,i)  = sample_trunc_normal(mean_v, var_v, beta(i), arma::datum::inf)(0);
    
    // Update alpha row after diagonal change
    alpha.row(i) = arma::sum(Omg.row(i)) - Omg.row(i);
    
    // -- Off-diagonal entries j > i --
    for (int j = i + 1; j < p; ++j) {
      double aij   = S(i,i)/N + S(j,j)/N + 1.0/(N * tau_sq);
      double bij   = -arma::sum(Omg.col(i) % S.col(j) / N +
                                Omg.col(j) % S.col(i) / N) +
                                Omg(i,j) * (S(i,i)/N + S(j,j)/N);
      double lower = -std::min(alpha(i,j), alpha(j,i));   // DD lower bound
      double cij   = 2.0 * q / (1.0 - q) *
        std::sqrt(1.0 / (N * aij * tau_sq)) *
        std::exp(N * bij * bij / (2.0 * aij)) *
        (arma::normcdf(-bij * std::sqrt((double)N / aij)) -
        arma::normcdf(-bij * std::sqrt((double)N / aij) +
        lower * std::sqrt((double)N * aij)));
      
      if (std::isinf(cij)) {
        bin = 1.0;
      } else {
        double q_star = cij / (1.0 + cij);
        bin = R::rbinom(1, q_star);
      }
      
      double entrynew;
      if (bin == 1.0) {
        arma::vec a(1);  a(0) = aij;
        entrynew = (sample_trunc_normal(bij / a, 1.0 / (N * a), lower, 0.0))(0);
      } else {
        entrynew = 0.0;
      }
      
      // -- Update alpha and beta bookkeeping --
      double delta = entrynew - Omg(i,j);
      if (delta != 0.0) {
        for (int kk = 0; kk < p; ++kk) {
          if (kk != i && kk != j) {
            alpha(i, kk) += delta;
            alpha(j, kk) += delta;
          }
        }
        beta(i) -= delta;
        beta(j) -= delta;
      }
      Omg(i,j) = Omg(j,i) = entrynew;
      
      if (post) {
        nz_count(i,j) = nz_count(j,i) = nz_count(i,j) + bin;
      }
    }
  }
  
  if (post) {
    Omg_sum  += Omg;
    nz_count += Ip;
  }
  
  arma::umat sp_iter = arma::conv_to<arma::umat>::from(Omg != 0);
  sp_iter.diag().zeros();
  return sp_iter;
}




