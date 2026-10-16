// =============================================================================
// AGIBBS_one_iter.cpp
// Single-iteration wrappers for A-Gibbs
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
// FUNCTION 1: one iteration of A-Gibbs  (Gibbs_Corrected — For M-matrix)
// =============================================================================
static arma::umat iter_A_Mmat(arma::mat& Omg,
                                  arma::mat& nz_count,
                                  arma::mat& Omg_sum,
                                  const arma::mat& S,
                                  const arma::mat& U,
                                  const arma::mat& A,
                                  const arma::mat& Ip,
                                  double k,
                                  int N, int p,
                                  double q, double r1, double s1,
                                  bool post) {
  
  arma::mat R(p, p);
  
  // -- Sample auxiliary R --
  if (N < p) {
    arma::mat U0 = reshape(std_multinorm(p * p), p, p);
    arma::mat W  = (U0 - U * (U.t() * U0 +
      A * reshape(std_multinorm(N * p), N, p))) * std::sqrt(k);
    R = W + k * (Omg - U * (U.t() * Omg)) + N * Ip;
  } else {
    R = reshape(std_multinorm(p * p), p, p) * arma::chol(A) + A * Omg + N * Ip;
  }
  
  // -- Sample diagonal --
  double scale_d = (N > p) ? 1.0 / k : 1.0 / (k + 0.01);
  Omg.diag() = sample_trunc_normal(diagvec(R) * scale_d,
           scale_d * arma::eye<arma::mat>(p, p),
           0.0, arma::datum::inf);
  
  // -- Sample global shrinkage --
  double eta_sq_gam = randg(distr_param(
    r1 + 0.5 * (arma::accu(Omg != 0) - p) / 2.0,
    1.0 / (s1 + (arma::accu(Omg % Omg) - arma::sum(Omg.diag() % Omg.diag())) / 2.0)));
  double k_star = 2.0 * k + eta_sq_gam;
  
  // -- Sample off-diagonal --
  double bin;
  for (int i = 0; i < p; ++i) {
    for (int j = i + 1; j < p; ++j) {
      double Rsum = R(i,j) + R(j,i);
      double cij  = 2.0 * q * std::exp(Rsum * Rsum / (2.0 * k_star)) *
        std::sqrt(eta_sq_gam) / ((1.0 - q) * std::sqrt(k_star)) *
        arma::normcdf(-Rsum / std::sqrt(k_star));
      
      if (std::isinf(cij)) {
        bin = 1.0;
      } else {
        double q_star = cij / (1.0 + cij);
        bin = R::rbinom(1, q_star);
      }
      
      if (bin == 1.0) {
        arma::vec ks(1);  ks(0) = k_star;
        Omg(i,j) = Omg(j,i) = (sample_trunc_normal(
          arma::vec(1).fill(Rsum / k_star), 1.0 / ks,
          R_NegInf, 0.0))(0);
      } else {
        Omg(i,j) = Omg(j,i) = 0.0;
      }
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

// =============================================================================
// FUNCTION 2: one iteration of A-Gibbs  (DAI_Gibbs — DD constraint)
// =============================================================================
static arma::umat iter_A_DD(arma::mat& Omg,
                            arma::mat& nz_count,
                            arma::mat& Omg_sum,
                            arma::mat& alpha,
                            arma::vec& beta,
                            const arma::mat& S,
                            const arma::mat& U,
                            const arma::mat& A,
                            const arma::mat& Ip,
                            double k,
                            int N, int p,
                            double q, double r1, double s1,
                            bool post) {
  
  arma::mat R(p, p);
  
  // -- Sample R --
  if (N < p) {
    arma::mat U0 = reshape(std_multinorm(p * p), p, p);
    arma::mat W  = (U0 - U * (U.t() * U0 +
      A * reshape(std_multinorm(N * p), N, p))) * std::sqrt(k);
    R = W + k * (Omg - U * (U.t() * Omg)) + N * Ip;
  } else {
    R = reshape(std_multinorm(p * p), p, p) * arma::chol(A) + A * Omg + N * Ip;
  }
  
  // -- Sample global shrinkage --
  double eta_sq_gam = randg(distr_param(
    r1 + 0.5 * (arma::accu(Omg != 0) - p) / 2.0,
    1.0 / (s1 + (arma::accu(Omg % Omg) - arma::sum(Omg.diag() % Omg.diag())) / 2.0)));
  double k_star = 2.0 * k + eta_sq_gam;
  
  // -- Off-diagonal: DD lower bound from alpha --
  double bin;
  for (int i = 0; i < p; ++i) {
    for (int j = i + 1; j < p; ++j) {
      double Rsum  = R(i,j) + R(j,i);
      double lower = -std::min(alpha(i,j), alpha(j,i));
      double cij   = 2.0 * q * std::exp(Rsum * Rsum / (2.0 * k_star)) *
        std::sqrt(eta_sq_gam) / ((1.0 - q) * std::sqrt(k_star)) *
        (arma::normcdf(-Rsum / std::sqrt(k_star)) -
        arma::normcdf(-Rsum / std::sqrt(k_star) +
        lower * std::sqrt(k_star)));
      
      if (std::isinf(cij)) {
        bin = 1.0;
      } else {
        double q_star = cij / (1.0 + cij);
        bin = R::rbinom(1, q_star);
      }
      
      double entrynew;
      if (bin == 1.0) {
        arma::vec ks(1);  ks(0) = k_star;
        entrynew = (sample_trunc_normal(
          arma::vec(1).fill(Rsum / k_star), 1.0 / ks, lower, 0.0))(0);
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
  
  // -- Diagonal: lower bound beta (DD constraint) --
  // Uses vectorised sample_trunc_normal_vec [5]
  arma::vec diag_new = sample_trunc_normal_vec(
    diagvec(R) / k,
    1.0 / k * arma::eye<arma::mat>(p, p),
    beta,
    arma::vec(p).fill(arma::datum::inf));
  Omg.diag() = diag_new;
  
  // Full recompute of alpha after diagonal update (mirrors DAI_Gibbs.cpp line 237/578) [5]
  alpha = arma::repmat(arma::sum(Omg, 1), 1, p) - Omg;
  
  if (post) {
    Omg_sum  += Omg;
    nz_count += Ip;
  }
  
  arma::umat sp_iter = arma::conv_to<arma::umat>::from(Omg != 0);
  sp_iter.diag().zeros();
  return sp_iter;
}
