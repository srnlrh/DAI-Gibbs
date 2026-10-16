// =============================================================================
// sample_multinormal.cpp
// Exports std_multinorm(int p), which draws a p-dimensional standard normal
// vector using the Ziggurat algorithm for fast normal variate generation.
// This serves as the core multivariate normal building block for the Gibbs
// samplers: callers scale and shift the output to obtain draws from N(mu, Sigma).
// Requires: RcppArmadillo, RcppZiggurat. The static Ziggurat instance must be
// seeded via set_seed_zigg() in seed.cpp before use.
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
arma::vec std_multinorm(int p) {
  arma::vec z(p);
  for(int i = 0; i < p; ++i){
    z[i] = zigg.norm();
  }
  return z;
}
