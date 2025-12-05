source("derivation.R")

library(assertthat)
library(tidyverse)

# helper
auc_to_d <- function(auc) {
  qnorm(auc) * sqrt(2)
}

# n samples
n <- 1e6

# AI
theta_m <- 0.3

# wrong
awm <- .1
bwm <- .5

# correct
acm <- .6
bcm <- .9

# Human
theta_h <- 0.6

# wrong
awh <- .3
bwh <- .6

# correct
ach <- .7
bch <- .8

# Monte Carlo samples
sample_mc <- function(n, theta, ac, bc, aw, bw) {
  y <- runif(n) < theta
  p <- ifelse(y,
    runif(n, ac, bc),
    runif(n, aw, bw)
  )
  list(
    p = p,
    y = y
  )
}

# get logit posterior
get_llr <- function(p, theta, ac, bc, aw, bw) {
  c <- dunif(p, ac, bc, log = TRUE)
  w <- dunif(p, aw, bw, log = TRUE)

  c - w + qlogis(theta)
}

# integrate AUC numerically
get_auc <- function(ac, bc, aw, bw) {
  f <- function(x) {
    # p(c > w)
    dunif(x, ac, bc) * punif(x, aw, bw)
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

  # infinity tie-breaker
  obs <- mean(c(ifelse(am > ah, ym, yh), ifelse(am < ah, yh, ym)))
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
  n, 0.6, 0.9, .3, .4, .1, .2, .7, .8, .5, .6,
  n, 0.6, 0.9, .7, .8, .5, .6, .3, .4, .1, .2,
  n, 0.6, 0.9, .5, .9, .3, .6, .2, .8, .4, .5,
  n, 0.6, 0.9, .2, .8, .4, .5, .5, .9, .3, .6,
  n, 0.6, 0.9, .3, .4, .1, .2, .2, .8, .4, .5,
  n, 0.6, 0.9, .7, .8, .5, .6, .5, .9, .3, .6,
  n, 0.6, 0.9, .5, .9, .3, .6, .7, .8, .5, .6,
  n, 0.6, 0.9, .2, .8, .4, .5, .3, .4, .1, .2,
  n, 0.6, 0.9, .5, .9, .3, .6, .7, .8, .5, .6,
  n, 0.6, 0.9, .2, .8, .4, .5, .3, .4, .1, .2,
  n, 0.6, 0.9, .3, .4, .1, .2, .2, .8, .4, .5,
  n, 0.6, 0.9, .7, .8, .5, .6, .5, .9, .3, .6,
  n, 0.03, 0.97, .3, .4, .1, .2, .7, .8, .5, .6,
  n, 0.03, 0.97, .7, .8, .5, .6, .3, .4, .1, .2,
  n, 0.03, 0.97, .5, .9, .3, .6, .2, .8, .4, .5,
  n, 0.03, 0.97, .2, .8, .4, .5, .5, .9, .3, .6,
  n, 0.03, 0.97, .3, .4, .1, .2, .2, .8, .4, .5,
  n, 0.03, 0.97, .7, .8, .5, .6, .5, .9, .3, .6,
  n, 0.03, 0.97, .5, .9, .3, .6, .7, .8, .5, .6,
  n, 0.03, 0.97, .2, .8, .4, .5, .3, .4, .1, .2,
  n, 0.03, 0.97, .5, .9, .3, .6, .7, .8, .5, .6,
  n, 0.03, 0.97, .2, .8, .4, .5, .3, .4, .1, .2,
  n, 0.03, 0.97, .3, .4, .1, .2, .2, .8, .4, .5,
  n, 0.03, 0.97, .7, .8, .5, .6, .5, .9, .3, .6,
  n, 0.2, 0.1, .3, .4, .1, .2, .7, .8, .5, .6,
  n, 0.2, 0.1, .7, .8, .5, .6, .3, .4, .1, .2,
  n, 0.2, 0.1, .5, .9, .3, .6, .2, .8, .4, .5,
  n, 0.2, 0.1, .2, .8, .4, .5, .5, .9, .3, .6,
  n, 0.2, 0.1, .3, .4, .1, .2, .2, .8, .4, .5,
  n, 0.2, 0.1, .7, .8, .5, .6, .5, .9, .3, .6,
  n, 0.2, 0.1, .5, .9, .3, .6, .7, .8, .5, .6,
  n, 0.2, 0.1, .2, .8, .4, .5, .3, .4, .1, .2,
  n, 0.2, 0.1, .5, .9, .3, .6, .7, .8, .5, .6,
  n, 0.2, 0.1, .2, .8, .4, .5, .3, .4, .1, .2,
  n, 0.2, 0.1, .3, .4, .1, .2, .2, .8, .4, .5,
  n, 0.2, 0.1, .7, .8, .5, .6, .5, .9, .3, .6,
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
