generate_data <- function(r, n, p, type, a = NULL, b = NULL, prob = 0.05,
                          lower = 2, upper = 5) {
  
  type <- match.arg(tolower(type), choices = c("line", "grid", "er"))
  
  if (type == "grid") {
    if (is.null(a) || is.null(b))
      stop("For type = 'grid', please supply both 'a' (nrow) and 'b' (ncol).")
    if (a * b != p)
      warning("a * b != p: grid dimensions may not tile exactly into p nodes.")
  }
  
  # Construct the Adjacency Matrix
  Adj <- switch(type,
                "line" = generate_line(p, lower = lower, upper = upper),
                "grid" = generate_grid(p, nrow = a, ncol = b, lower = lower, upper = upper),
                "er"   = generate_ER(p, prob = prob, lower = lower, upper = upper)
  )
  
  # Construct the precision matrix
  # Shift diagonal so Theta is positive definite (M-matrix construction)
  del       <- 1.05 * max(eigen(Adj)$values)
  Theta     <- del * diag(p) - Adj
  
  # Standardize to get a correlation-like precision matrix
  E         <- diag(sqrt(diag(solve(Theta))))
  Omega_true <- E %*% Theta %*% E
  
  # Generate r datasets of size n x p
  Sigma <- solve(Omega_true)   # covariance matrix
  L     <- t(chol(Sigma))      # lower Cholesky factor
  
  data_list <- vector("list", r)
  for (j in seq_len(r)) {
    data_list[[j]] <- t(L %*% matrix(rnorm(n * p), p, n))  # n x p matrix
  }
  
  ## Return
  return(list(
    Omega_true = Omega_true,
    data_list  = data_list
  ))
}