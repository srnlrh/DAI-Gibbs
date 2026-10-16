# Bayesian Graphical Models under Positivity Constraints

Implementation code for the methods proposed in:

> **Bayesian Graphical Models under Positivity Constraints: A Scalable Generalized Likelihood Approach**

This repository provides efficient posterior sampling algorithms for sparse precision matrix estimation in Gaussian graphical models under M-matrix (total positivity) constraints, using a generalized Bayesian framework based on the D-trace loss with spike-and-slab priors.

------------------------------------------------------------------------

## Methods

Three Gibbs samplers are implemented:

| Sampler | Description |
|------------------------------------|------------------------------------|
| **D-Gibbs** | Componentwise Direct Gibbs Sampler |
| **A-Gibbs** | Augmented Gibbs Sampler (induces conditional independence via matrix-variate augmentation) |
| **I-Gibbs** | Interweaving Gibbs Sampler (alternates D and A updates; recommended) |

Both M-matrix and diagonal-dominance (DD) constrained variants are supported via the `DD` flag.

------------------------------------------------------------------------

## Repository Structure

``` text

├── seed.cpp #RNG seed setters (Armadillo + Ziggurat) 
├── sample_multinormal.cpp #Standard multivariate normal sampler 
├── sample_truncated_normal.cpp #Truncated normal sampler (scalar and vector bounds) 
├── DGIBBS_one_iter.cpp #Single D-Gibbs iteration (M-matrix and DD) 
├── AGIBBS_one_iter.cpp #Single A-Gibbs iteration (M-matrix and DD) 
├── Gibbs_sampler.cpp #Master run_gibbs() function (D / A / I) 
├── Estimate.R #R wrapper: estimate_precision() 
├── grid_search.R #Hyperparameter q selection via grid search 
├── compute_metrics.R #Evaluation: MCC, F-score, Relative Error, Prop. PD 
├── modularity.R #Modularity score for financial network evaluation 
└── Data_Generation/ 
  ├── Line.R #Line graph adjacency generator 
  ├── Grid.R #Grid graph adjacency generator 
  ├── Erdos_Renyi.R #Erdős-Rényi graph adjacency generator 
  ├── Generate_data.R #Synthetic data generator (calls graph generators) 
  ├── generate_financial_data.R #S&P 500 log-return data download and processing 
```

------------------------------------------------------------------------

## Dependencies

### R packages

``` r
install.packages(c( "Rcpp", "RcppArmadillo", "RcppZiggurat", "RcppTN", 
                         "MASS", "Matrix", "igraph", "quantmod", "rvest", 
                         "dplyr", "tidyr", "purrr", "openxlsx" ))
```

A C++ compiler compatible with C++11 is required (standard with most R installations via Rtools on Windows or Xcode on macOS).

### Usage

## 1. Compile C++ sources

``` r
library(Rcpp); library(RcppArmadillo)
library(RcppZiggurat); library(RcppTN) 
library(MASS); library(Matrix)

sourceCpp("seed.cpp") 
sourceCpp("sample_multinormal.cpp") 
sourceCpp("sample_truncated_normal.cpp") 
sourceCpp("DGIBBS_one_iter.cpp") 
sourceCpp("AGIBBS_one_iter.cpp") 
sourceCpp("Gibbs_sampler.cpp")

set.seed(1234); set_seed_arma(1234); set_seed_zigg(1234)
```

## 2. Generate synthetic data

``` r
source("Data_Generation/Generate_data.R") 
source("Data_Generation/Line.R") 
source("Data_Generation/Grid.R") 
source("Data_Generation/Erdos_Renyi.R")

# Example: Line graph, p=50, n=100, 50 replicates
out <- generate_data(r = 50, n = 100, p = 50, type = "line") 
Omega_true <- out$Omega_true
data_list  <- out$data_list
```

## 3. Select Hyperparameter via Grid Search

``` r
source("grid_search.R")

# Use 4/5 of observations for training, 
# 1/5 for validation (as in the paper)

train <- data_list[[1]][1:80, ]
test <- data_list[[1]][81:100, ]

q_opt <- grid_search( test = test, train = train, qvals = c(1/50, 1/50\^1.5, 1/50\^2), sampler = "I" )
```

## 4. Estimate the Precision Matrix

``` r
source("Estimate.R")

Omega_hat <- estimate_precision( X = train, q = q_opt, Omega_init = diag(50), burnin = 2000, nmc = 1000, sampler = "I" \# "D", "A", or "I" )
```

## 5. Evaluate performance

``` r
source("compute_metrics.R")

metrics <- compute_metrics( Omega_true = Omega_true, Omega_hat = Omega_hat, norm_type = "F" \# Frobenius norm, as reported in the paper \# Pass samples = obj\$samples[,,(burnin+1):(burnin+nmc)] for prop_PD )

print(metrics) # $MCC, $F_score, $rel_error, $prop_PD
```

## 6. Generate Financial Time-Series data

``` r
source("Data_Generation/generate_financial_data.R")

fin_data <- generate_financial_data()
```

Hyperparameter Choices Following Section 4.4 of the paper:

- $q$ is selected via grid search over $\{1/p, 1/p^{1.5}, 1/p^2\}$ using a held-out test set.
- $\tau^2$ is treated as random with an Inverse-Gamma(1e-4, 1e-8) prior (fully Bayesian).
- For I-Gibbs, a D-step is invoked every period = 100 iterations.
- Default burn-in: 2000 iterations; post-burn-in samples: 1000.

Other competing methods has been implemented using existing packages or code shared by authors:

- GLASSO is implemented using packages `glasso` and `CVglasso`.
- CLIME is implemented using package `clime`.
- GGL and DDGL is implemented using the code shared by the authors. It can be found [here](https://github.com/STAC-USC/Graph_Learning).
