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
void set_seed_arma(int seed){
  arma::arma_rng::set_seed(seed);
}

// [[Rcpp::export]]
void set_seed_zigg(int seed){
  zigg.setSeed(seed);
}