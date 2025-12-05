source("derivation.R")

library(assertthat)
library(tidyverse)

library(extraDistr)

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
theta_m <- 0.3

# wrong
awm <- 10
bwm <- 20

# correct
acm <- 21
bcm <- 10

# Human
theta_h <- 0.9

# wrong
awh <- 30
bwh <- 20

# correct
ach <- 50
bch <- 10

# Monte Carlo samples
sample_mc <- function(n, theta, ac, bc, aw, bw) {
  y <- runif(n) < theta
  p <- ifelse(y,
    rkumar(n, ac, bc),
    rkumar(n, aw, bw)
  )
  list(
    p = p,
    y = y
  )
}

# get logit posterior
get_llr <- function(p, theta, ac, bc, aw, bw) {
  p <- clamp(p, 1e-5, 1 - 1e-5)

  c <- dkumar(p, ac, bc, log = TRUE)
  w <- dkumar(p, aw, bw, log = TRUE)

  c - w + qlogis(theta)
}

# integrate AUC numerically
get_auc <- function(ac, bc, aw, bw) {
  f <- function(x) {
    # p(c > w)
    dkumar(x, ac, bc) * pkumar(x, aw, bw)
  }
  integrate(f, lower = 0, upper = 1)$value
}

# simulate combined accuracy
sim_com_acc <- function(n, theta_m, theta_h,
                        acm, bcm, awm, bwm,
                        ach, bch, awh, bwh) {
  sm <- sample_mc(n, theta_m, acm, bcm, awm, bwm)
  sh <- sample_mc(n, theta_h, ach, bch, awh, bwh)

  ym <- sm$y
  yh <- sh$y

  pm <- sm$p
  ph <- sh$p

  am <- get_llr(pm, theta_m, acm, bcm, awm, bwm)
  ah <- get_llr(ph, theta_h, ach, bch, awh, bwh)

  obs <- mean(ifelse(am > ah, ym, yh))
  obs
}


# get mean absolute error
get_mae <- function(n, theta_m, theta_h,
                    acm, bcm, awm, bwm,
                    ach, bch, awh, bwh) {
  # theoretical combined accuracy
  auc_m <- get_auc(acm, bcm, awm, bwm)
  auc_h <- get_auc(ach, bch, awh, bwh)

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
    acm, bcm, awm, bwm,
    ach, bch, awh, bwh
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
  acm, bcm, awm, bwm,
  ach, bch, awh, bwh
)

df <- tribble(
  ~n, ~theta_m, ~theta_h, ~acm, ~bcm, ~awm, ~bwm, ~ach, ~bch, ~awh, ~bwh,
  n, 0.6, 0.9, 2, 1, 1, 2, 2, 1, 1, 2,
  n, 0.5, 0.3, 2, 1, 1, 2, 2, 1, 1, 2,
  n, 0.6, 0.9, 5, 2, 1, 2, 5, 2, 1, 2,
  n, 0.5, 0.3, 5, 2, 1, 2, 5, 2, 1, 2,
  n, 0.6, 0.9, 5, 2, 1, 3, 4, 2, 2, 3,
  n, 0.5, 0.3, 5, 2, 1, 3, 4, 2, 2, 3,
  n, 0.6, 0.9, 1, .2, .1, 3, 4, .2, .2, 3,
  n, 0.9, 0.6, 1, .2, .1, 3, 4, .2, .2, 3,
  n, 0.5, 0.3, 1, .2, .1, 3, 4, .2, .2, 3,
  n, 0.3, 0.5, 1, .2, .1, 3, 4, .2, .2, 3,
  n, 0.1, 0.9, 2, 1, 1, 2, 2, 1, 1, 2,
  n, 0.5, 0.7, 2, 1, 1, 2, 2, 1, 1, 2,
  n, 0.1, 0.9, 5, 2, 1, 2, 5, 2, 1, 2,
  n, 0.5, 0.7, 5, 2, 1, 2, 5, 2, 1, 2,
  n, 0.1, 0.9, 5, 2, 1, 3, 4, 2, 2, 3,
  n, 0.5, 0.7, 5, 2, 1, 3, 4, 2, 2, 3,
  n, 0.1, 0.9, 1, .2, .1, 3, 4, .2, .2, 3,
  n, 0.9, 0.1, 1, .2, .1, 3, 4, .2, .2, 3,
  n, 0.5, 0.7, 1, .2, .1, 3, 4, .2, .2, 3,
  n, 0.7, 0.5, 1, .2, .1, 3, 4, .2, .2, 3,
)

df <- df |>
  rowwise() |>
  mutate(res = list(get_mae(
    n, theta_m, theta_h,
    acm, bcm, awm, bwm,
    ach, bch, awh, bwh
  ))) |>
  unnest_wider(res) |>
  arrange(desc(mae)) |>
  glimpse()
