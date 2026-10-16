library(quantmod)
library(rvest)
library(dplyr)
library(tidyr)
library(purrr)
library(openxlsx)

# -----------------------------------------------------------------------------
# generate_financial_data
#
# Downloads daily adjusted closing prices for S&P 500 stocks from five GICS
# sectors (Jan 2022 - Dec 2023), computes scaled log-returns, and returns a
# matrix of returns with a sector label column appended.
#
# Arguments:
#   none — data source and date range are fixed to match the paper's experiment.
#
# Returns:
#   A numeric matrix of dimension (p x (n+1)), where:
#     - rows correspond to stocks (p = 286)
#     - columns t1...tn are scaled log-returns (n = 503 trading days)
#     - the final column "sector" is a numeric sector label (1–5)
#   Row names are stock ticker symbols.
# -----------------------------------------------------------------------------
generate_financial_data <- function() {
  
  # --- Scrape S&P 500 constituent list and sector labels from Wikipedia -------
  url <- "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
  
  sp500 <- url %>%
    read_html() %>%
    html_table(fill = TRUE) %>%
    .[[1]] %>%
    select(Symbol, Security, `GICS Sector`)
  
  # Yahoo Finance uses "-" instead of "." in tickers (e.g. BRK.B -> BRK-B)
  sp500$Symbol <- gsub("\\.", "-", sp500$Symbol)
  
  # --- Filter to the five target sectors used in the paper -------------------
  target_sectors <- c(
    "Information Technology",
    "Health Care",
    "Financials",
    "Consumer Discretionary",
    "Consumer Staples"
  )
  
  sector_stocks <- sp500 %>%
    filter(`GICS Sector` %in% target_sectors)
  
  # --- Download adjusted closing prices from Yahoo Finance -------------------
  get_prices <- function(sym) {
    tryCatch({
      x <- getSymbols(sym,
                      src          = "yahoo",
                      from         = "2022-01-01",
                      to           = "2023-12-31",
                      auto.assign  = FALSE)
      tibble(
        date           = index(x),
        symbol         = sym,
        adjusted_close = as.numeric(Ad(x))
      )
    }, error = function(e) NULL)
  }
  
  price_data_1 <- map_dfr(sector_stocks$Symbol, get_prices)
  
  # --- Identify and remove problematic/delisted tickers -----------------------
  # A ticker is considered problematic if any of the following are true:
  #   1. It failed to download entirely (not in price_data_1 at all)
  #   2. It has fewer trading days than the median (likely delisted mid-period)
  #   3. It has any NA in its adjusted close price series
  
  # Count trading days per ticker
  days_per_ticker <- price_data_1 %>%
    group_by(symbol) %>%
    summarise(n_days    = n(),
              has_na    = any(is.na(adjusted_close)),
              .groups   = "drop")
  
  # Expected number of trading days: use median across all tickers as reference
  expected_days <- median(days_per_ticker$n_days)
  
  # Flag tickers with missing data or substantially fewer observations
  # Threshold: allow up to 5 missing days to tolerate minor gaps (e.g. holidays)
  bad_tickers <- days_per_ticker %>%
    filter(has_na | n_days < expected_days - 5) %>%
    pull(symbol)
  
  if (length(bad_tickers) > 0) {
    message(sprintf("Removing %d problematic/delisted ticker(s): %s",
                    length(bad_tickers),
                    paste(bad_tickers, collapse = ", ")))
  }
  
  # Remove all rows belonging to bad tickers
  price_data <- price_data_1 %>%
    filter(!symbol %in% bad_tickers)
  
  # --- Pivot to wide format: rows = dates, columns = tickers -----------------
  price_matrix <- price_data %>%
    pivot_wider(
      names_from  = symbol,
      values_from = adjusted_close
    ) %>%
    arrange(date)
  
  price_mat <- price_matrix %>%
    select(-date) %>%
    as.matrix()                       # 501 x 286
  
  # --- Compute scaled log-returns --------------------------------------------
  # X_{i,j} = 100 * log(P_{i+1,j} / P_{i,j}), as defined in the paper
  log_returns <- 100 * diff(log(price_mat))   # 500 x 286
  return_mat  <- t(log_returns)               # 286 x 500
  
  # --- Align sector labels to matched tickers --------------------------------
  ss_un <- unique(sector_stocks$Symbol)
  pd_un <- unique(price_data$symbol)
  s1    <- setdiff(seq_along(ss_un), which(ss_un %in% pd_un))
  sector_stocks_mod <- sector_stocks[-s1, ]
  
  sector_vec <- sector_stocks_mod %>%
    distinct(Symbol, `GICS Sector`) %>%
    arrange(match(Symbol, colnames(price_mat))) %>%
    pull(`GICS Sector`)
  
  sector_numeric <- as.numeric(factor(sector_vec))
  
  # --- Assemble final matrix with sector column ------------------------------
  final_matrix <- cbind(return_mat, sector = sector_numeric)
  
  rownames(final_matrix) <- colnames(price_mat)
  colnames(final_matrix) <- c(
    paste0("t", seq_len(ncol(final_matrix) - 1)),
    "sector"
  )
  
  return(final_matrix)
}