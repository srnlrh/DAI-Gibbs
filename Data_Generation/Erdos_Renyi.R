generate_ER <- function(p, prob = 0.05, lower = 2, upper = 5){
  Adj <- matrix(0, p, p)
  idx <- upper.tri(Adj)
  Adj[idx] <- rbinom(sum(idx), 1, prob) * runif(sum(idx), lower, upper)
  Adj <- Adj + t(Adj)
  return(Adj)
}

