source("derivation.R")

library(assertthat)
library(tidyverse)
library(logitnorm)

# helper
auc_to_d <- function(auc) {
  qnorm(auc) * sqrt(2)
}

# Define clamp function
clamp <- function(x, lower, upper) {
  pmin(pmax(x, lower), upper)
}

# n samples
n <- 1e5

# AI
theta_m <- 0.5

# wrong
mwm <- 1
swm <- 10

# correct
mcm <- 5
scm <- 2

# Human
theta_h <- 0.7

# wrong
mwh <- 1
swh <- 5

# correct
mch <- 2
sch <- 2

# Monte Carlo samples
sample_mc <- function(n, theta, mc, sc, mw, sw) {
  y <- runif(n) < theta
  p <- ifelse(y,
    rlogitnorm(n, mc, sc),
    rlogitnorm(n, mw, sw)
  )
  list(
    p = p,
    y = y
  )
}

# get logit posterior
get_llr <- function(p, theta, mc, sc, mw, sw) {
  p <- clamp(p, 1e-5, 1 - 1e-5)

  c <- dlogitnorm(p, mc, sc, log = TRUE)
  w <- dlogitnorm(p, mw, sw, log = TRUE)

  c - w + qlogis(theta)
}

# integrate AUC numerically
get_auc <- function(mc, sc, mw, sw) {
  f <- function(x) {
    # p(c > w)
    dlogitnorm(x, mc, sc) * plogitnorm(x, mw, sw)
  }
  integrate(f, lower = 0, upper = 1)$value
}

# simulate combined accuracy
sim_com_acc <- function(n, theta_m, theta_h,
                        mcm, scm, mwm, swm,
                        mch, sch, mwh, swh) {
  sm <- sample_mc(n, theta_m, mcm, scm, mwm, swm)
  sh <- sample_mc(n, theta_h, mch, sch, mwh, swh)

  ym <- sm$y
  yh <- sh$y

  pm <- sm$p
  ph <- sh$p

  am <- get_llr(pm, theta_m, mcm, scm, mwm, swm)
  ah <- get_llr(ph, theta_h, mch, sch, mwh, swh)

  obs <- mean(ifelse(am > ah, ym, yh))
  obs
}


# get mean absolute error
get_mae <- function(n, theta_m, theta_h,
                    mcm, scm, mwm, swm,
                    mch, sch, mwh, swh) {
  # theoretical combined accuracy
  auc_m <- get_auc(mcm, scm, mwm, swm)
  auc_h <- get_auc(mch, sch, mwh, swh)

  assert_that(auc_m >= 0.5 && auc_h >= 0.5)

  d_m <- auc_to_d(auc_m)
  d_h <- auc_to_d(auc_h)

  pre <- combined_accuracy_d(
    theta_m = theta_m, theta_h = theta_h,
    d_m = d_m, d_h = d_h
  )

  # simulate combined accuracy
  obs <- sim_com_acc(
    n, theta_m, theta_h,
    mcm, scm, mwm, swm,
    mch, sch, mwh, swh
  )

  list(
    auc_m = auc_m, auc_h = auc_h,
    d_m = d_m, d_h = d_h,
    pre = pre, obs = obs,
    mae = abs(obs - pre), pct = abs(obs - pre) / pre * 100
  )
}

get_mae(
  n, theta_m, theta_h,
  mcm, scm, mwm, swm,
  mch, sch, mwh, swh
)

df <- tribble(
  ~n, ~theta_m, ~theta_h, ~mcm, ~scm, ~mwm, ~swm, ~mch, ~sch, ~mwh, ~swh,
  n, 0.6, 0.9, 2, 3, 1, 5, 2, 3, 1, 5,
  n, 0.5, 0.3, 2, 3, 1, 5, 2, 3, 1, 5,
  n, 0.6, 0.9, 5, 3, 1, 2, 5, 2, 1, 3,
  n, 0.5, 0.3, 5, 3, 1, 2, 5, 2, 1, 3,
  n, 0.6, 0.9, 5, 2, 1, 3, 4, 2, 2, 3,
  n, 0.5, 0.3, 5, 2, 1, 3, 4, 2, 2, 3,
  n, 0.6, 0.9, 1, .2, .1, 3, 4, .2, .2, 3,
  n, 0.9, 0.6, 1, .2, .1, 3, 4, .2, .2, 3,
  n, 0.5, 0.3, 1, .2, .1, 3, 4, .2, .2, 3,
  n, 0.3, 0.5, 1, .2, .1, 3, 4, .2, .2, 3,
  n, 0.1, 0.9, 2, 3, 1, 5, 2, 3, 1, 5,
  n, 0.5, 0.7, 2, 3, 1, 5, 2, 3, 1, 5,
  n, 0.1, 0.9, 5, 3, 1, 2, 5, 2, 1, 3,
  n, 0.5, 0.7, 5, 3, 1, 2, 5, 2, 1, 3,
  n, 0.1, 0.9, 5, 2, 1, 3, 4, 2, 2, 3,
  n, 0.5, 0.7, 5, 2, 1, 3, 4, 2, 2, 3,
  n, 0.1, 0.9, 1, .2, .1, 3, 4, .2, .2, 3,
  n, 0.9, 0.1, 1, .2, .1, 3, 4, .2, .2, 3,
  n, 0.5, 0.7, 1, .2, .1, 3, 4, .2, .2, 3,
  n, 0.7, 0.5, 1, .2, .1, 3, 4, .2, .2, 3,
  n, 0.6, 0.9, 2, 3, -1, 5, 2, 3, -1, 5,
  n, 0.5, 0.3, 2, 3, -1, 5, 2, 3, -1, 5,
  n, 0.6, 0.9, -1, 2, -5, 3, -1, 2, -5, 3,
  n, 0.5, 0.3, -1, 2, -5, 3, -1, 2, -5, 3,
  n, 0.6, 0.9, -1, 3, -5, 2, -2, 3, -4, 2,
  n, 0.5, 0.3, -1, 3, -5, 2, -2, 3, -4, 2,
  n, 0.6, 0.9, -.1, 3, -1, .2, -.2, 3, -4, .2,
  n, 0.9, 0.6, -.1, 3, -1, .2, -.2, 3, -4, .2,
  n, 0.5, 0.3, -.1, 3, -1, .2, -.2, 3, -4, .2,
  n, 0.3, 0.5, -.1, 3, -1, .2, -.2, 3, -4, .2,
  n, 0.1, 0.9, 2, 3, -1, 5, 2, 3, -1, 5,
  n, 0.5, 0.7, 2, 3, -1, 5, 2, 3, -1, 5,
  n, 0.1, 0.9, -1, 2, -5, 3, -1, 2, -5, 3,
  n, 0.5, 0.7, -1, 2, -5, 3, -1, 2, -5, 3,
  n, 0.1, 0.9, -1, 3, -5, 2, -2, 3, -4, 2,
  n, 0.5, 0.7, -1, 3, -5, 2, -2, 3, -4, 2,
  n, 0.1, 0.9, -.1, 3, -1, .2, -.2, 3, -4, .2,
  n, 0.9, 0.1, -.1, 3, -1, .2, -.2, 3, -4, .2,
  n, 0.5, 0.7, -.1, 3, -1, .2, -.2, 3, -4, .2,
  n, 0.7, 0.5, -.1, 3, -1, .2, -.2, 3, -4, .2,
)

df <- df |>
  rowwise() |>
  mutate(res = list(get_mae(
    n, theta_m, theta_h,
    mcm, scm, mwm, swm,
    mch, sch, mwh, swh
  ))) |>
  unnest_wider(res) |>
  arrange(desc(mae)) |>
  glimpse()
