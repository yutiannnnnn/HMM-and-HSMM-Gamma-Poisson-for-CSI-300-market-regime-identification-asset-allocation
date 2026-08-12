# HSMM replication-style script with in-sample comparison and out-of-sample backtest
# Fixed version: consistent benchmark settings, corrected Poisson AIC option,
# segment-based state-band plots, diagnostic outputs, unified benchmark/performance conventions,
# and explicit turnover diagnostics for out-of-sample strategies.
#
# Usage for this hard-coded version:
#   1. Put this script and CSI300_2005_2026_returns.xlsx in the same folder.
#   2. Run this script directly in RStudio or with:
#        Rscript hsmm_CSI300_2005_2026_hardcoded.R
#
# Input file must contain a date column and a close-price column.

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(ggplot2)
  library(mhsmm)
  library(tibble)
  library(tidyr)
  library(purrr)
  library(tools)
})

# =========================
# 1) Config
# =========================
# HARD-CODED INPUT VERSION
# Put this R file and the Excel file below in the same folder, then run this script directly.
# This version deliberately ignores command-line arguments to avoid accidental inputs
# from RStudio/LiveView/other IDE helpers.

DATA_FILE_NAME <- "CSI300_2005_2026_returns.xlsx"

# Optional: if the data file is not in the same folder as this script, paste the
# absolute path here. Example on macOS:
# DATA_FILE_PATH <- "/Users/zhangyutian/Downloads/CSI300_2005_2026_returns.xlsx"
DATA_FILE_PATH <- ""

OUT_DIR <- "results_hsmm_CSI300_2005_2026"
INITIAL_WINDOW_END_OVERRIDE <- as.Date(NA)

get_script_dir <- function() {
  # Works when executed with source() or Rscript in most environments.
  this_file <- tryCatch(normalizePath(sys.frame(1)$ofile), error = function(e) NA_character_)
  if (!is.na(this_file) && nzchar(this_file)) return(dirname(this_file))

  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- sub("^--file=", "", file_arg[1])
    return(dirname(normalizePath(script_path, mustWork = FALSE)))
  }

  getwd()
}

find_input_file <- function(file_name, explicit_path = "") {
  explicit_path <- trimws(explicit_path)

  # 1. User-provided absolute or relative path has highest priority.
  if (nzchar(explicit_path)) {
    explicit_path <- path.expand(explicit_path)
    if (file.exists(explicit_path)) {
      return(normalizePath(explicit_path, mustWork = TRUE))
    }
    stop(paste0("DATA_FILE_PATH was set, but the file does not exist:\n", explicit_path))
  }

  script_dir <- get_script_dir()
  home_dir <- path.expand("~")
  search_dirs <- unique(c(
    getwd(),
    script_dir,
    file.path(home_dir, "Downloads"),
    file.path(home_dir, "Desktop"),
    file.path(home_dir, "Documents")
  ))
  search_dirs <- search_dirs[dir.exists(search_dirs)]

  # 2. Exact filename search in common folders.
  exact_candidates <- unique(c(
    file.path(search_dirs, file_name),
    file_name
  ))
  exact_candidates <- exact_candidates[file.exists(exact_candidates)]
  if (length(exact_candidates) > 0) {
    return(normalizePath(exact_candidates[1], mustWork = TRUE))
  }

  # 3. Flexible search for likely CSI 300 Excel/CSV files in common folders.
  all_candidates <- unlist(lapply(search_dirs, function(d) {
    list.files(
      d,
      pattern = "\\.(xlsx|xls|csv)$",
      full.names = TRUE,
      recursive = FALSE,
      ignore.case = TRUE
    )
  }), use.names = FALSE)
  all_candidates <- unique(all_candidates[file.exists(all_candidates)])

  preferred <- all_candidates[grepl("csi\\s*[-_ ]?300|hs300|沪深|300", basename(all_candidates), ignore.case = TRUE)]
  if (length(preferred) == 1) {
    return(normalizePath(preferred[1], mustWork = TRUE))
  }

  # 4. In RStudio/interactive sessions, open a file picker instead of failing.
  if (interactive()) {
    message("Could not automatically find ", file_name, ". Please choose the CSI 300 Excel/CSV file manually.")
    chosen <- tryCatch(file.choose(), error = function(e) NA_character_)
    if (!is.na(chosen) && nzchar(chosen) && file.exists(chosen)) {
      return(normalizePath(chosen, mustWork = TRUE))
    }
  }

  hint <- if (length(all_candidates) > 0) {
    paste0("\nNearby CSV/Excel files found:\n", paste(utils::head(all_candidates, 20), collapse = "\n"))
  } else {
    ""
  }

  stop(paste0(
    "Cannot find the data file: ", file_name, "\n",
    "Current working directory: ", getwd(), "\n",
    "Script directory: ", script_dir, "\n",
    "Fix: either put the data file in one of these folders, or set DATA_FILE_PATH near the top of this script.",
    hint
  ))
}

SCRIPT_DIR <- get_script_dir()
EXCEL_FILE <- find_input_file(DATA_FILE_NAME, DATA_FILE_PATH)
cat("Using input file:", EXCEL_FILE, "\n")
cat("Using output folder:", OUT_DIR, "\n")

STATE_LIST <- 2:7
N_RANDOM_STARTS <- 8
N_RANDOM_STARTS_ROLLING <- 2
MAXIT <- 120
M_MAX <- 260
SET_SEED <- 123

# Out-of-sample settings. Keep the same values in the HMM script if you compare HMM and HSMM.
INITIAL_WINDOW_N <- 1260       # about five trading years when no date override is supplied
REFIT_FREQ <- 21               # monthly re-estimation/rebalancing frequency in trading days
ROLLING_MV_WINDOW <- 252       # rolling benchmark window
RISK_AVERSION <- 6
MAX_LEVERAGE <- 1.5
TRADING_COST_BPS <- 10         # 10 bps per unit of absolute turnover
RISK_FREE_DAILY <- 0           # set to zero if no risk-free daily series is available

# IMPORTANT AIC convention:
# Poisson duration is treated as a one-parameter duration distribution per state.
# The shift is fixed rather than estimated, so the Poisson duration penalty is +K.
# If you deliberately estimate or tune a state-specific shift, set this to TRUE.
COUNT_POISSON_SHIFT_AS_PARAMETER <- FALSE
POISSON_SHIFT_FIXED <- 1L

set.seed(SET_SEED)
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

# =========================
# 2) Data preparation
# =========================
parse_date_safe <- function(x) {
  if (inherits(x, "Date")) return(x)
  if (inherits(x, "POSIXct") || inherits(x, "POSIXlt")) return(as.Date(x))
  if (is.numeric(x)) return(as.Date(x, origin = "1899-12-30"))
  x_chr <- trimws(as.character(x))
  out <- suppressWarnings(as.Date(x_chr))
  if (all(is.na(out))) out <- suppressWarnings(as.Date(x_chr, format = "%Y%m%d"))
  if (all(is.na(out))) out <- suppressWarnings(as.Date(x_chr, format = "%d/%m/%Y"))
  out
}

parse_numeric_safe <- function(x) {
  as.numeric(gsub(",", "", as.character(x)))
}

load_price_data <- function(path) {
  if (!file.exists(path)) stop(paste("File not found:", path))

  ext <- tolower(file_ext(path))
  if (ext == "csv") {
    df <- read.csv(path, stringsAsFactors = FALSE)
  } else if (ext %in% c("xls", "xlsx")) {
    df <- read_excel(path)
  } else {
    stop("Unsupported file format. Please provide a CSV or Excel file.")
  }

  names(df) <- tolower(trimws(names(df)))

  date_candidates <- c("trade_date", "date", "datetime", "time", "day")
  close_candidates <- c("close", "adj_close", "adjclose", "adjusted", "price", "close_price")

  date_col <- intersect(date_candidates, names(df))[1]
  close_col <- intersect(close_candidates, names(df))[1]

  if (is.na(date_col)) stop("Data must contain a date column, e.g. trade_date, date, or datetime.")
  if (is.na(close_col)) stop("Data must contain a close-price column, e.g. close, adj_close, or adjclose.")

  out <- df %>%
    transmute(
      date = parse_date_safe(.data[[date_col]]),
      close = parse_numeric_safe(.data[[close_col]])
    ) %>%
    arrange(date) %>%
    distinct(date, .keep_all = TRUE) %>%
    filter(!is.na(date), !is.na(close), close > 0)

  out$log_return <- c(NA_real_, diff(log(out$close)))
  out$simple_return <- c(NA_real_, diff(out$close) / head(out$close, -1))
  out <- out %>% filter(!is.na(log_return), !is.na(simple_return))

  if (nrow(out) < 300) stop("Too few valid observations after cleaning.")
  out
}

to_hsmm_data <- function(x) {
  obj <- list(x = as.numeric(x), N = length(x))
  class(obj) <- "hsmm.data"
  obj
}

# =========================
# 3) Helpers
# =========================
runs_from_states <- function(states) {
  states <- as.integer(states)
  if (length(states) == 0) {
    return(data.frame(state = integer(), start = integer(), end = integer(), length = integer()))
  }
  starts <- c(1, which(diff(states) != 0) + 1)
  ends <- c(starts[-1] - 1, length(states))
  data.frame(state = states[starts], start = starts, end = ends, length = ends - starts + 1)
}

correctly_decoded_states <- function(ret, states, state_means) {
  rr <- as.numeric(ret)
  states <- as.integer(states)
  runs <- runs_from_states(states)
  if (nrow(runs) == 0) return(NA_real_)
  eps <- 1e-10
  ok <- 0
  neutral_band <- stats::sd(rr, na.rm = TRUE)
  for (i in seq_len(nrow(runs))) {
    s <- runs$state[i]
    a <- runs$start[i]
    b <- runs$end[i]
    seg_sum <- sum(rr[a:b], na.rm = TRUE)
    mu <- state_means[s]
    if (is.na(mu)) next
    if (mu > eps && seg_sum > 0) ok <- ok + 1
    if (mu < -eps && seg_sum < 0) ok <- ok + 1
    if (abs(mu) <= eps && abs(seg_sum) <= neutral_band) ok <- ok + 1
  }
  ok / nrow(runs)
}

regime_summary <- function(ret, states, model_name, k) {
  rr <- as.numeric(ret)
  states <- as.integer(states)
  runs <- runs_from_states(states)
  state_ids <- sort(unique(states))
  rows <- lapply(state_ids, function(s) {
    idx <- which(states == s)
    r_s <- rr[idx]
    runs_s <- runs %>% filter(state == s)
    tibble(
      model = model_name,
      k_states = k,
      state = s,
      mean_daily = mean(r_s, na.rm = TRUE),
      sd_daily = sd(r_s, na.rm = TRUE),
      mean_daily_pct = mean(r_s, na.rm = TRUE) * 100,
      sd_daily_pct = sd(r_s, na.rm = TRUE) * 100,
      sharpe_daily = ifelse(sd(r_s, na.rm = TRUE) > 0, mean(r_s, na.rm = TRUE) / sd(r_s, na.rm = TRUE), NA_real_),
      mean_ann_pct = mean(r_s, na.rm = TRUE) * 252 * 100,
      vol_ann_pct = sd(r_s, na.rm = TRUE) * sqrt(252) * 100,
      proportion = length(idx) / length(rr),
      proportion_pct = 100 * length(idx) / length(rr),
      occurrences = nrow(runs_s),
      total_duration = sum(runs_s$length),
      avg_duration = mean(runs_s$length),
      min_duration = min(runs_s$length),
      max_duration = max(runs_s$length)
    )
  })
  bind_rows(rows)
}

count_params_hmm <- function(k) {
  k * (k - 1) + (k - 1) + 2 * k
}

count_params_hsmm_poisson <- function(k) {
  transition_params <- k * (k - 2)       # embedded transition matrix has zero diagonal
  init_params <- k - 1
  emission_params <- 2 * k               # Gaussian mean and standard deviation
  duration_params <- if (COUNT_POISSON_SHIFT_AS_PARAMETER) 2 * k else k
  transition_params + init_params + emission_params + duration_params
}

count_params_hsmm_gamma <- function(k) {
  transition_params <- k * (k - 2)
  init_params <- k - 1
  emission_params <- 2 * k
  duration_params <- 2 * k               # shape and scale
  transition_params + init_params + emission_params + duration_params
}

compute_aic <- function(loglik, n_params) {
  2 * n_params - 2 * loglik
}

random_start_values <- function(ret, k, seed) {
  set.seed(seed)
  qs <- quantile(ret, probs = seq(0.1, 0.9, length.out = k), na.rm = TRUE)
  ret_sd <- max(sd(ret, na.rm = TRUE), 1e-4)
  mu0 <- as.numeric(qs + rnorm(k, sd = ret_sd * 0.1))
  sigma0 <- rep(ret_sd, k) * runif(k, 0.6, 1.4)

  trans_hmm <- matrix(runif(k * k, 0.1, 1), nrow = k)
  trans_hmm <- trans_hmm / rowSums(trans_hmm)

  trans_hsmm <- matrix(runif(k * k, 0.1, 1), nrow = k)
  diag(trans_hsmm) <- 0
  trans_hsmm <- trans_hsmm / rowSums(trans_hsmm)

  init <- rep(1 / k, k)

  list(
    init = init,
    trans_hmm = trans_hmm,
    trans_hsmm = trans_hsmm,
    mu = mu0,
    sigma = sigma0,
    lambda = pmax(round(runif(k, 8, 60)), 1),
    shift = rep(POISSON_SHIFT_FIXED, k),
    shape = runif(k, 1.2, 8),
    scale = runif(k, 2, 20)
  )
}

fit_one_hmm <- function(train_obj, ret, k, seed) {
  st <- random_start_values(ret, k, seed)
  spec <- hmmspec(
    init = st$init,
    trans = st$trans_hmm,
    parms.emission = list(mu = st$mu, sigma = st$sigma),
    dens.emission = dnorm.hsmm
  )
  fit <- hmmfit(train_obj, spec, mstep = mstep.norm, maxit = MAXIT)
  pred <- predict(fit, train_obj, method = "viterbi")
  loglik <- tail(fit$loglik, 1)
  means <- fit$model$parms.emission$mu
  states <- as.integer(pred$s)

  tibble(
    model = "HMM", k_states = k, seed = seed, loglik = loglik,
    n_params = count_params_hmm(k),
    AIC = compute_aic(loglik, count_params_hmm(k)),
    CD = correctly_decoded_states(ret, states, means),
    fit_object = list(fit), state_path = list(states), state_means = list(means)
  )
}

fit_one_hsmm_poisson <- function(train_obj, ret, k, seed) {
  st <- random_start_values(ret, k, seed)
  spec <- hsmmspec(
    init = st$init,
    transition = st$trans_hsmm,
    parms.emission = list(mu = st$mu, sigma = st$sigma),
    sojourn = list(lambda = st$lambda, shift = st$shift, type = "poisson"),
    dens.emission = dnorm.hsmm
  )
  fit <- hsmmfit(train_obj, spec, mstep = mstep.norm, M = M_MAX, maxit = MAXIT)
  pred <- predict(fit, train_obj, method = "viterbi")
  loglik <- tail(fit$loglik, 1)
  means <- fit$model$parms.emission$mu
  states <- as.integer(pred$s)

  tibble(
    model = "HSMM_Poisson", k_states = k, seed = seed, loglik = loglik,
    n_params = count_params_hsmm_poisson(k),
    AIC = compute_aic(loglik, count_params_hsmm_poisson(k)),
    CD = correctly_decoded_states(ret, states, means),
    fit_object = list(fit), state_path = list(states), state_means = list(means)
  )
}

fit_one_hsmm_gamma <- function(train_obj, ret, k, seed) {
  st <- random_start_values(ret, k, seed)
  spec <- hsmmspec(
    init = st$init,
    transition = st$trans_hsmm,
    parms.emission = list(mu = st$mu, sigma = st$sigma),
    sojourn = list(shape = st$shape, scale = st$scale, type = "gamma"),
    dens.emission = dnorm.hsmm
  )
  fit <- hsmmfit(train_obj, spec, mstep = mstep.norm, M = M_MAX, maxit = MAXIT)
  pred <- predict(fit, train_obj, method = "viterbi")
  loglik <- tail(fit$loglik, 1)
  means <- fit$model$parms.emission$mu
  states <- as.integer(pred$s)

  tibble(
    model = "HSMM_Gamma", k_states = k, seed = seed, loglik = loglik,
    n_params = count_params_hsmm_gamma(k),
    AIC = compute_aic(loglik, count_params_hsmm_gamma(k)),
    CD = correctly_decoded_states(ret, states, means),
    fit_object = list(fit), state_path = list(states), state_means = list(means)
  )
}

rank_with_ar <- function(df) {
  df %>%
    filter(is.finite(AIC), is.finite(CD)) %>%
    group_by(model, k_states) %>%
    mutate(
      rank_AIC = rank(AIC, ties.method = "min"),
      rank_CD = rank(-CD, ties.method = "min"),
      AR = (rank_AIC + rank_CD) / 2
    ) %>%
    arrange(model, k_states, AR, AIC) %>%
    ungroup()
}

state_intervals <- function(date, states) {
  states <- as.integer(states)
  runs <- runs_from_states(states)
  step <- suppressWarnings(median(as.numeric(diff(date)), na.rm = TRUE))
  if (!is.finite(step) || step <= 0) step <- 1
  runs %>%
    mutate(
      xmin = date[start],
      xmax = date[end] + step,
      state = factor(state)
    )
}

plot_state_bands <- function(date, price, states, title_txt, file) {
  tmp <- tibble(date = date, price = price)
  intervals <- state_intervals(date, states)
  p <- ggplot(tmp, aes(x = date, y = price)) +
    geom_rect(
      data = intervals,
      aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf, fill = state),
      inherit.aes = FALSE,
      alpha = 0.18
    ) +
    geom_line(linewidth = 0.45) +
    labs(title = title_txt, x = "Date", y = "Price", fill = "State") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")
  ggsave(file, p, width = 11, height = 5, dpi = 160)
}

# =========================
# 4) Out-of-sample backtest helpers
# =========================
get_last_state_probs <- function(fit, train_obj, model_type) {
  # For HSMM, mhsmm does not expose a simple one-step filtered probability object here.
  # Therefore the last Viterbi state from the training window is converted into a one-hot vector.
  # This is strictly based on information available up to the training-window end.
  if (model_type == "HMM") {
    pred <- predict(fit, train_obj, method = "smoothing")
    probs <- pred$p
    return(as.numeric(probs[nrow(probs), ]))
  }

  pred <- predict(fit, train_obj, method = "viterbi")
  states <- as.integer(pred$s)
  k <- length(fit$model$parms.emission$mu)
  probs <- rep(0, k)
  last_state <- tail(states[!is.na(states)], 1)
  if (length(last_state) == 1 && last_state >= 1 && last_state <= k) probs[last_state] <- 1
  probs
}

expected_moments <- function(probs_T, means, variances) {
  probs_T <- as.numeric(probs_T)
  means <- as.numeric(means)
  variances <- as.numeric(variances)
  exp_ret <- sum(probs_T * means)
  second_moment <- sum(probs_T * (variances + means^2))
  exp_var <- max(second_moment - exp_ret^2, 1e-8)
  list(mean = exp_ret, var = exp_var)
}

optimal_weight <- function(forecast_mean, forecast_var, gamma = RISK_AVERSION) {
  if (!is.finite(forecast_mean) || !is.finite(forecast_var) || forecast_var <= 0) return(1.0)
  w <- (1 / gamma) * (forecast_mean / forecast_var)
  as.numeric(pmax(0, pmin(MAX_LEVERAGE, w)))
}

rolling_empirical_weight <- function(ret_window, gamma = RISK_AVERSION) {
  ret_window <- as.numeric(ret_window)
  ret_window <- ret_window[is.finite(ret_window)]
  if (length(ret_window) < 10) return(1.0)
  mu <- mean(ret_window)
  v <- var(ret_window)
  if (!is.finite(v) || v <= 0) v <- 1e-8
  w <- (1 / gamma) * (mu / v)
  as.numeric(pmax(0, pmin(MAX_LEVERAGE, w)))
}

compute_monthly_rolling_mv_weights <- function(data, test_idx) {
  # Benchmark 2 convention, matched to the HMM Python script:
  #   * estimate the empirical mean-variance signal from the previous 252 trading days;
  #   * use log_return for the signal estimation;
  #   * rebalance only every REFIT_FREQ trading days;
  #   * keep the target weight unchanged between rebalancing dates.
  # Portfolio returns are still computed from simple_return in make_strategy_series().
  if (length(test_idx) == 0) return(numeric(0))

  weights <- numeric(length(test_idx))
  i <- 1L
  while (i <= length(test_idx)) {
    current_abs_idx <- test_idx[i]
    hist_start <- max(1L, current_abs_idx - ROLLING_MV_WINDOW)
    hist_end <- current_abs_idx - 1L

    if (hist_end >= hist_start) {
      hist_ret <- data$log_return[hist_start:hist_end]
      w <- rolling_empirical_weight(hist_ret)
    } else {
      w <- 1.0
    }

    block_end <- min(i + REFIT_FREQ - 1L, length(test_idx))
    weights[i:block_end] <- w
    i <- block_end + 1L
  }

  weights
}

make_strategy_series <- function(weights, returns, dates, name, trading_cost_bps = TRADING_COST_BPS, initial_weight = 1.0) {
  n <- min(length(weights), length(returns), length(dates))
  weights <- as.numeric(weights[seq_len(n)])
  returns <- as.numeric(returns[seq_len(n)])
  dates <- dates[seq_len(n)]

  weights[!is.finite(weights)] <- 1.0
  returns[!is.finite(returns)] <- 0

  rf <- RISK_FREE_DAILY
  gross_ret <- weights * returns + (1 - weights) * rf
  # Charge transaction costs on the first out-of-sample day if the first target
  # weight differs from the initial benchmark exposure. This matches the Python
  # HMM script's convention of starting from a 100% risky-asset exposure.
  turnover <- c(abs(weights[1] - initial_weight), abs(diff(weights)))

  # Scheduled rebalancing indicator. The first out-of-sample observation is a
  # rebalancing point, and the target weight is then updated every REFIT_FREQ
  # trading days. Transaction costs are charged only on these rebalancing dates.
  is_rebalance <- ((seq_len(n) - 1L) %% REFIT_FREQ == 0L)
  turnover[!is_rebalance] <- 0

  cost <- turnover * (trading_cost_bps / 10000)
  net_ret <- gross_ret - cost

  tibble(
    date = dates,
    strategy = name,
    weight = weights,
    risky_return = returns,
    gross_return = gross_ret,
    turnover = turnover,
    is_rebalance = is_rebalance,
    cost = cost,
    net_return = net_ret,
    wealth_gross = cumprod(1 + gross_ret),
    wealth_net = cumprod(1 + net_ret)
  )
}

evaluate_strategy_series <- function(series_tbl) {
  x <- series_tbl
  n <- nrow(x)
  ann_factor <- 252
  final_net <- tail(x$wealth_net, 1)
  final_gross <- tail(x$wealth_gross, 1)
  ann_return_cagr <- final_net^(ann_factor / n) - 1
  mean_ann_return <- mean(x$net_return, na.rm = TRUE) * ann_factor
  sd_ret <- sd(x$net_return, na.rm = TRUE) * sqrt(ann_factor)
  sharpe_net <- if (sd_ret > 0) ann_return_cagr / sd_ret else NA_real_

  gross_cagr <- final_gross^(ann_factor / n) - 1
  gross_vol <- sd(x$gross_return, na.rm = TRUE) * sqrt(ann_factor)
  sharpe_gross <- if (gross_vol > 0) gross_cagr / gross_vol else NA_real_

  drawdown <- x$wealth_net / cummax(x$wealth_net) - 1

  # Turnover diagnostics. The daily average is reported for completeness, while
  # turnover per rebalancing date is the clearest execution metric for the thesis
  # tables because the strategies are rebalanced monthly.
  if (!"is_rebalance" %in% names(x)) {
    x$is_rebalance <- ((seq_len(n) - 1L) %% REFIT_FREQ == 0L)
  }
  rebalance_turnover <- x$turnover[x$is_rebalance]
  avg_turnover_daily <- mean(x$turnover, na.rm = TRUE)
  avg_turnover_per_rebalance <- mean(rebalance_turnover, na.rm = TRUE)
  ann_turnover <- avg_turnover_per_rebalance * (252 / REFIT_FREQ)

  tibble(
    strategy = unique(x$strategy)[1],
    start_date = min(x$date),
    end_date = max(x$date),
    n_obs = n,
    ann_return_cagr_pct = ann_return_cagr * 100,
    mean_ann_pct = mean_ann_return * 100,
    vol_ann_pct = sd_ret * 100,
    sharpe_net = sharpe_net,
    sharpe_gross = sharpe_gross,
    max_drawdown_pct = min(drawdown, na.rm = TRUE) * 100,
    final_wealth_net = final_net,
    final_wealth_gross = final_gross,
    avg_exposure_pct = mean(x$weight, na.rm = TRUE) * 100,
    avg_turnover_daily_pct = avg_turnover_daily * 100,
    avg_turnover_per_rebalance_pct = avg_turnover_per_rebalance * 100,
    ann_turnover_pct = ann_turnover * 100
  )
}

evaluate_strategy <- function(weights, returns, dates, name, trading_cost_bps = TRADING_COST_BPS) {
  series_tbl <- make_strategy_series(weights, returns, dates, name, trading_cost_bps)
  evaluate_strategy_series(series_tbl)
}

backtest_model <- function(data, model_type, k, initial_window_end) {
  test_idx <- which(data$date > initial_window_end)
  if (length(test_idx) == 0) stop("No out-of-sample data after initial window.")
  if (sum(data$date <= initial_window_end) < 60) stop("Initial training window too short.")

  weights <- numeric(length(test_idx))
  dates_test <- data$date[test_idx]
  returns_test <- data$simple_return[test_idx]

  last_fit <- NULL
  last_probs <- NULL
  last_means <- NULL
  last_vars <- NULL

  for (i in seq_along(test_idx)) {
    current_date <- dates_test[i]
    train_data <- data %>% filter(date < current_date)
    train_ret <- train_data$log_return

    if (i == 1 || (i - 1) %% REFIT_FREQ == 0) {
      train_obj <- to_hsmm_data(train_ret)
      fit_fun <- switch(
        model_type,
        HMM = fit_one_hmm,
        HSMM_Poisson = fit_one_hsmm_poisson,
        HSMM_Gamma = fit_one_hsmm_gamma,
        stop("Unsupported model_type")
      )

      candidates <- map(seq_len(N_RANDOM_STARTS_ROLLING), function(sid) {
        tryCatch(
          fit_fun(train_obj, train_ret, k, seed = SET_SEED + 100000 + 1000 * k + 10 * i + sid),
          error = function(e) NULL
        )
      }) %>% compact()

      if (length(candidates) > 0) {
        cand_tbl <- bind_rows(candidates)
        fit_result <- cand_tbl %>% arrange(AIC) %>% slice(1)
        last_fit <- fit_result$fit_object[[1]]
        last_probs <- get_last_state_probs(last_fit, train_obj, model_type)
        last_means <- fit_result$state_means[[1]]
        last_vars <- (last_fit$model$parms.emission$sigma)^2
      } else if (is.null(last_fit)) {
        weights[i] <- 1.0
        next
      }
    }

    if (!is.null(last_fit)) {
      moments <- expected_moments(last_probs, last_means, last_vars)
      weights[i] <- optimal_weight(moments$mean, moments$var)
    } else {
      weights[i] <- 1.0
    }
  }

  list(weights = weights, dates = dates_test, returns = returns_test)
}

plot_wealth_curves <- function(series_tbl, file) {
  p <- ggplot(series_tbl, aes(x = date, y = wealth_net, color = strategy)) +
    geom_line(linewidth = 0.45) +
    labs(title = "Out-of-Sample Cumulative Wealth", x = "Date", y = "Cumulative Wealth") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")
  ggsave(file, p, width = 12, height = 6, dpi = 160)
}

# =========================
# 5) Main: in-sample and out-of-sample
# =========================
cat("1/6 Loading data...\n")
price_data <- load_price_data(EXCEL_FILE)
cat("Data range:", as.character(min(price_data$date)), "to", as.character(max(price_data$date)), "\n")

if (!is.na(INITIAL_WINDOW_END_OVERRIDE)) {
  INITIAL_WINDOW_END <- INITIAL_WINDOW_END_OVERRIDE
} else if (nrow(price_data) > INITIAL_WINDOW_N) {
  INITIAL_WINDOW_END <- price_data$date[INITIAL_WINDOW_N]
} else {
  INITIAL_WINDOW_END <- price_data$date[floor(nrow(price_data) * 0.2)]
}

if (INITIAL_WINDOW_END <= min(price_data$date) || INITIAL_WINDOW_END >= max(price_data$date)) {
  stop("Initial window end must be inside the data range.")
}
cat("Initial window end:", as.character(INITIAL_WINDOW_END), "\n")

train_obj_full <- to_hsmm_data(price_data$log_return)

write.csv(
  data.frame(
    n_obs = nrow(price_data),
    start_date = min(price_data$date),
    end_date = max(price_data$date),
    initial_window_end = INITIAL_WINDOW_END,
    test_start = min(price_data$date[price_data$date > INITIAL_WINDOW_END]),
    test_end = max(price_data$date),
    mean_log_daily = mean(price_data$log_return, na.rm = TRUE),
    sd_log_daily = sd(price_data$log_return, na.rm = TRUE),
    mean_log_daily_pct = 100 * mean(price_data$log_return, na.rm = TRUE),
    sd_log_daily_pct = 100 * sd(price_data$log_return, na.rm = TRUE),
    mean_simple_daily = mean(price_data$simple_return, na.rm = TRUE),
    sd_simple_daily = sd(price_data$simple_return, na.rm = TRUE),
    mean_simple_daily_pct = 100 * mean(price_data$simple_return, na.rm = TRUE),
    sd_simple_daily_pct = 100 * sd(price_data$simple_return, na.rm = TRUE)
  ),
  file.path(OUT_DIR, "data_overview.csv"),
  row.names = FALSE
)

settings_tbl <- tibble(
  setting = c(
    "state_list", "n_random_starts", "n_random_starts_rolling", "maxit", "m_max",
    "initial_window_n", "initial_window_end", "refit_freq", "rolling_mv_window",
    "risk_aversion", "max_leverage", "trading_cost_bps", "risk_free_daily",
    "poisson_shift_fixed", "count_poisson_shift_as_parameter"
  ),
  value = c(
    paste(STATE_LIST, collapse = ","), N_RANDOM_STARTS, N_RANDOM_STARTS_ROLLING, MAXIT, M_MAX,
    INITIAL_WINDOW_N, as.character(INITIAL_WINDOW_END), REFIT_FREQ, ROLLING_MV_WINDOW,
    RISK_AVERSION, MAX_LEVERAGE, TRADING_COST_BPS, RISK_FREE_DAILY,
    POISSON_SHIFT_FIXED, COUNT_POISSON_SHIFT_AS_PARAMETER
  )
)
write.csv(settings_tbl, file.path(OUT_DIR, "backtest_settings.csv"), row.names = FALSE)

cat("2/6 Running HSMM multi-start in-sample...\n")
all_fits <- list()
for (k in STATE_LIST) {
  for (seed in seq_len(N_RANDOM_STARTS)) {
    cat(sprintf("  K=%d, seed=%d\n", k, seed))
    all_fits[[length(all_fits) + 1]] <- tryCatch(
      fit_one_hsmm_poisson(train_obj_full, price_data$log_return, k, seed),
      error = function(e) { message("HSMM_Poisson failed: ", e$message); NULL }
    )
    all_fits[[length(all_fits) + 1]] <- tryCatch(
      fit_one_hsmm_gamma(train_obj_full, price_data$log_return, k, seed),
      error = function(e) { message("HSMM_Gamma failed: ", e$message); NULL }
    )
  }
}

fits_tbl <- bind_rows(all_fits)
if (nrow(fits_tbl) == 0) stop("No in-sample model converged.")

ranked_tbl <- rank_with_ar(fits_tbl)
write.csv(
  ranked_tbl %>% select(model, k_states, seed, loglik, n_params, AIC, CD, rank_AIC, rank_CD, AR),
  file.path(OUT_DIR, "all_model_starts_ranked.csv"),
  row.names = FALSE
)

best_tbl <- ranked_tbl %>%
  group_by(model, k_states) %>%
  slice_min(order_by = AR, n = 1, with_ties = FALSE) %>%
  ungroup

write.csv(
  best_tbl %>% select(model, k_states, seed, loglik, n_params, AIC, CD, AR),
  file.path(OUT_DIR, "in_sample_model_comparison_hsmm.csv"),
  row.names = FALSE
)

cat("3/6 Writing regime summaries and segment-based state-band plots...\n")
regime_rows <- list()
for (i in seq_len(nrow(best_tbl))) {
  one_row <- best_tbl[i, ]
  states <- unlist(one_row$state_path[[1]])
  regime_rows[[i]] <- regime_summary(price_data$log_return, states, one_row$model, one_row$k_states)
  title_txt <- paste0("In-Sample State Bands: ", one_row$model, " (K=", one_row$k_states, ")")
  fn <- paste0("state_bands_", one_row$model, "_k", one_row$k_states, ".png")
  plot_state_bands(price_data$date, price_data$close, states, title_txt, file.path(OUT_DIR, fn))
}
regime_tbl <- bind_rows(regime_rows)
write.csv(regime_tbl, file.path(OUT_DIR, "in_sample_regime_summary_hsmm.csv"), row.names = FALSE)

cat("4/6 Running out-of-sample backtests...\n")
test_data <- price_data %>% filter(date > INITIAL_WINDOW_END)
if (nrow(test_data) == 0) stop("No out-of-sample data after initial window.")

series_list <- list()
perf_list <- list()

bh_weights <- rep(1.0, nrow(test_data))
bh_series <- make_strategy_series(bh_weights, test_data$simple_return, test_data$date, "Benchmark_BuyHold")
series_list[[length(series_list) + 1]] <- bh_series
perf_list[[length(perf_list) + 1]] <- evaluate_strategy_series(bh_series)

test_idx <- which(price_data$date > INITIAL_WINDOW_END)
roll_weights <- compute_monthly_rolling_mv_weights(price_data, test_idx)
roll_series <- make_strategy_series(
  roll_weights,
  test_data$simple_return,
  test_data$date,
  "Benchmark_RollingMV_monthly_log_signal"
)
series_list[[length(series_list) + 1]] <- roll_series
perf_list[[length(perf_list) + 1]] <- evaluate_strategy_series(roll_series)

model_types <- c("HSMM_Poisson", "HSMM_Gamma")
for (model in model_types) {
  for (k in STATE_LIST) {
    cat(sprintf("  Backtesting %s K=%d\n", model, k))
    bt <- tryCatch(
      backtest_model(price_data, model, k, INITIAL_WINDOW_END),
      error = function(e) { message("Backtest failed: ", e$message); NULL }
    )
    if (!is.null(bt)) {
      strategy_name <- paste0(model, "_K", k)
      model_series <- make_strategy_series(bt$weights, bt$returns, bt$dates, strategy_name)
      series_list[[length(series_list) + 1]] <- model_series
      perf_list[[length(perf_list) + 1]] <- evaluate_strategy_series(model_series)
    }
  }
}

all_series <- bind_rows(series_list)
all_perf <- bind_rows(perf_list)

write.csv(all_series, file.path(OUT_DIR, "out_of_sample_strategy_series.csv"), row.names = FALSE)
write.csv(all_perf, file.path(OUT_DIR, "out_of_sample_performance.csv"), row.names = FALSE)
plot_wealth_curves(all_series, file.path(OUT_DIR, "out_of_sample_wealth_curves.png"))

cat("5/6 Generating performance summary...\n")
perf_summary <- all_perf %>%
  mutate(model_group = case_when(
    grepl("Benchmark", strategy) ~ "Benchmark",
    grepl("HSMM_Poisson", strategy) ~ "HSMM_Poisson",
    grepl("HSMM_Gamma", strategy) ~ "HSMM_Gamma",
    TRUE ~ strategy
  )) %>%
  select(
    strategy, model_group, start_date, end_date, n_obs,
    ann_return_cagr_pct, mean_ann_pct, vol_ann_pct,
    sharpe_net, sharpe_gross, max_drawdown_pct,
    final_wealth_net, final_wealth_gross,
    avg_exposure_pct,
    avg_turnover_daily_pct,
    avg_turnover_per_rebalance_pct,
    ann_turnover_pct
  )

write.csv(perf_summary, file.path(OUT_DIR, "performance_summary_table.csv"), row.names = FALSE)

cat("6/6 Done.\n")
cat("Outputs saved to:", normalizePath(OUT_DIR, mustWork = FALSE), "\n")

