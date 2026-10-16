library(igraph)

compute_modularity <- function(Omega, groups) {
  
  # Internal workhorse: modularity for a single matrix 
  .modularity_single <- function(mat, grps) {
    
    p <- nrow(mat)
    
    # Validate that Omega is square
    if (ncol(mat) != p)
      stop("Each Omega matrix must be square (nrow == ncol).")
    
    # Validate groups length and type
    if (length(grps) != p)
      stop("'groups' must have length equal to nrow(Omega) = ncol(Omega).")
    if (!is.factor(grps))
      stop("'groups' must be a factor.")
    
    # Build binary adjacency from non-zero off-diagonal entries of Omega
    adj <- (mat != 0) * 1L
    diag(adj) <- 0L
    
    # Warn if the resulting graph has isolated nodes
    if (any(rowSums(adj) == 0))
      warning("One or more nodes have no edges; modularity may be unreliable.")
    
    graph <- graph_from_adjacency_matrix(adj, mode = "undirected")
    
    # modularity() expects integer membership starting from 1
    Q <- modularity(graph, membership = as.integer(grps))
    
    return(Q)
  }
  
  # Dispatch: single matrix or list of matrices
  if (is.matrix(Omega)) {
    
    return(.modularity_single(Omega, groups))
    
  } else if (is.list(Omega)) {
    
    # Validate that every element is a matrix
    if (!all(sapply(Omega, is.matrix)))
      stop("When 'Omega' is a list, every element must be a matrix.")
    
    Q_vec <- sapply(seq_along(Omega), function(i) {
      .modularity_single(Omega[[i]], groups)
    })
    
    names(Q_vec) <- if (!is.null(names(Omega))) names(Omega) else paste0("Omega_", seq_along(Omega))
    return(Q_vec)
    
  } else {
    stop("'Omega' must be either a single matrix or a list of matrices.")
  }
}