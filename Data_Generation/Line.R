generate_line <- function(p, lower, upper){
  Adj_values <- runif(p-1, lower, upper)
  Adj <- as.matrix(bandSparse(p, k = c(-1, 0, 1), diagonals = list(Adj_values, rep(0, p), Adj_values)))
  return(Adj)
}