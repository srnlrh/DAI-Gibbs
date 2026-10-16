// =============================================================================
// sample_truncated_normal.cpp
// Exports two functions for sampling from (componentwise) truncated normals:
//   - sample_trunc_normal(): scalar bounds (lower, upper) applied to all p components.
//   - sample_trunc_normal_vec(): element-wise bounds vectors (lower[i], upper[i]).
// Both delegate to the R-level rtn() function from RcppTN, using only the diagonal
// of the covariance matrix (i.e., assumes a diagonal covariance input).
// Used throughout DGIBBS and AGIBBS to sample diagonal and off-diagonal entries
// of Omega subject to positivity or M-matrix sign constraints.
// =============================================================================


#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include <Ziggurat.h>
// [[Rcpp::depends(RcppZiggurat)]]

#include <RcppArmadilloExtensions/sample.h>
#include <algorithm>

using namespace Rcpp;
using namespace arma;

static Ziggurat::Ziggurat::Ziggurat zigg;

// [[Rcpp::plugins("cpp11")]]

// [[Rcpp::export]]
arma::vec sample_trunc_normal(const arma::vec& mean, const arma::mat& cov, double lower, double upper) {
  Rcpp::Function rtn1("rtn");
  int n = cov.n_cols;
  NumericVector mean_vec = Rcpp::wrap(mean);
  NumericVector sd_vec = Rcpp::wrap(arma::sqrt(cov.diag()));
  NumericVector lower_vec(n, lower);
  NumericVector upper_vec(n, upper);
  NumericVector samples = rtn1(mean_vec, sd_vec, lower_vec, upper_vec);
  return Rcpp::as<arma::vec>(samples);
}

// [[Rcpp::export]]
arma::vec sample_trunc_normal_vec(const arma::vec& mean, const arma::mat& cov,
                                  const arma::vec& lower, const arma::vec& upper) {
  Rcpp::Function rtn1("rtn");
  int n = cov.n_cols;
  NumericVector mean_vec  = Rcpp::wrap(mean);
  NumericVector sd_vec    = Rcpp::wrap(arma::sqrt(cov.diag()));
  NumericVector lower_vec = Rcpp::wrap(lower);
  NumericVector upper_vec = Rcpp::wrap(upper);
  NumericVector samples = rtn1(mean_vec, sd_vec, lower_vec, upper_vec);
  return Rcpp::as<arma::vec>(samples);
}

