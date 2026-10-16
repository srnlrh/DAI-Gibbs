generate_grid <- function(p, nrow, ncol, lower = 2, upper = 5){
  Adj <- matrix(0, p, p)
  for(i in 1:(p - 1)){
    if (i%%nrow == 0)
      Adj[i, i+nrow] = runif(1, 2, 5)
    else if (i <= (p-1) & i >= nrow * (ncol-1) + 1)
      Adj[i, i+1] = runif(1, 2, 5)
    else{
      Adj[i, i+1] = runif(1, 2, 5)
      Adj[i, i+nrow] = runif(1, 2, 5)
    }
  }
  Adj = Adj + t(Adj)
  return(Adj)
}



