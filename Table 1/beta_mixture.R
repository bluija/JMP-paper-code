source("derivation.R")

library(assertthat)
library(tidyverse)

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
wmw1 <- .4
awm1 <- 1
bwm1 <- 6

awm2 <- 5
bwm2 <- 4

# correct
cmw1 <- .6
acm1 <- 1
bcm1 <- .2

acm2 <- 9
bcm2 <- 8


# Human
theta_h <- 0.5

# wrong
whw1 <- .5
awh1 <- 2
bwh1 <- 10

awh2 <- 10
bwh2 <- 3

# correct
chw1 <- .2
ach1 <- 3
bch1 <- 10

ach2 <- 10
bch2 <- 2


# Mixture of beta
rbeta_mix <- function(n, w1, a1, b1, a2, b2) {
  y <- runif(n) < w1
  p <- ifelse(y,
    rbeta(n, a1, b1),
    rbeta(n, a2, b2)
  )
}

dbeta_mix <- function(x, w1, a1, b1, a2, b2, log = FALSE) {
  d1 <- dbeta(x, a1, b1, log = log)
  d2 <- dbeta(x, a2, b2, log = log)

  if (log) {
    log(w1 * exp(d1) + (1 - w1) * exp(d2))
  } else {
    w1 * d1 + (1 - w1) * d2
  }
}

pbeta_mix <- function(q, w1, a1, b1, a2, b2) {
  w1 * pbeta(q, a1, b1) + (1 - w1) * pbeta(q, a2, b2)
}

curve(dbeta_mix(x, cmw1, acm1, bcm1, acm2, bcm2), from = 0, to = 1)
curve(dbeta_mix(x, wmw1, awm1, bwm1, awm2, bwm2), from = 0, to = 1)
curve(dbeta_mix(x, chw1, ach1, bch1, ach2, bch2), from = 0, to = 1)
curve(dbeta_mix(x, whw1, awh1, bwh1, awh2, bwh2), from = 0, to = 1)


# Monte Carlo samples
sample_mc <- function(n, theta, c1, ac1, bc1, ac2, bc2, w1, aw1, bw1, aw2, bw2) {
  y <- runif(n) < theta
  p <- ifelse(y,
    rbeta_mix(n, c1, ac1, bc1, ac2, bc2),
    rbeta_mix(n, w1, aw1, bw1, aw2, bw2)
  )
  list(
    p = p,
    y = y
  )
}

# get logit posterior
get_llr <- function(p, theta, c1, ac1, bc1, ac2, bc2, w1, aw1, bw1, aw2, bw2) {
  p <- clamp(p, 1e-5, 1 - 1e-5)

  c <- dbeta_mix(p, c1, ac1, bc1, ac2, bc2, log = TRUE)
  w <- dbeta_mix(p, w1, aw1, bw1, aw2, bw2, log = TRUE)

  c - w + qlogis(theta)
}

# integrate AUC numerically
get_auc <- function(c1, ac1, bc1, ac2, bc2, w1, aw1, bw1, aw2, bw2) {
  f <- function(x) {
    # p(c > w)
    dbeta_mix(x, c1, ac1, bc1, ac2, bc2) * pbeta_mix(x, w1, aw1, bw1, aw2, bw2)
  }
  integrate(f, lower = 0, upper = 1)$value
}

# simulate combined accuracy
sim_com_acc <- function(n, theta_m, theta_h,
                        cmw1, acm1, bcm1, acm2, bcm2,
                        wmw1, awm1, bwm1, awm2, bwm2,
                        chw1, ach1, bch1, ach2, bch2,
                        whw1, awh1, bwh1, awh2, bwh2) {
  sm <- sample_mc(n, theta_m, cmw1, acm1, bcm1, acm2, bcm2, wmw1, awm1, bwm1, awm2, bwm2)
  sh <- sample_mc(n, theta_h, chw1, ach1, bch1, ach2, bch2, whw1, awh1, bwh1, awh2, bwh2)

  ym <- sm$y
  yh <- sh$y

  pm <- sm$p
  ph <- sh$p

  am <- get_llr(pm, theta_m, cmw1, acm1, bcm1, acm2, bcm2, wmw1, awm1, bwm1, awm2, bwm2)
  ah <- get_llr(ph, theta_h, chw1, ach1, bch1, ach2, bch2, whw1, awh1, bwh1, awh2, bwh2)

  obs <- mean(ifelse(am > ah, ym, yh))
  obs
}


# get mean absolute error
get_mae <- function(n, theta_m, theta_h,
                    cmw1, acm1, bcm1, acm2, bcm2,
                    wmw1, awm1, bwm1, awm2, bwm2,
                    chw1, ach1, bch1, ach2, bch2,
                    whw1, awh1, bwh1, awh2, bwh2) {
  # theoretical combined accuracy
  auc_m <- get_auc(cmw1, acm1, bcm1, acm2, bcm2, wmw1, awm1, bwm1, awm2, bwm2)
  auc_h <- get_auc(chw1, ach1, bch1, ach2, bch2, whw1, awh1, bwh1, awh2, bwh2)

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
    cmw1, acm1, bcm1, acm2, bcm2,
    wmw1, awm1, bwm1, awm2, bwm2,
    chw1, ach1, bch1, ach2, bch2,
    whw1, awh1, bwh1, awh2, bwh2
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
  cmw1, acm1, bcm1, acm2, bcm2,
  wmw1, awm1, bwm1, awm2, bwm2,
  chw1, ach1, bch1, ach2, bch2,
  whw1, awh1, bwh1, awh2, bwh2
)


df <- tribble(
  ~n, ~theta_m, ~theta_h,
  ~cmw1, ~acm1, ~bcm1, ~acm2, ~bcm2,
  ~wmw1, ~awm1, ~bwm1, ~awm2, ~bwm2,
  ~chw1, ~ach1, ~bch1, ~ach2, ~bch2,
  ~whw1, ~awh1, ~bwh1, ~awh2, ~bwh2,
  n, 0.6, 0.9,
  .6, 2, .2, 9, 8,
  .4, 1, 6, 5, 4,
  .2, 3, 10, 8, 2,
  .5, 2, 8, 6, 5,
  n, 0.1, 0.3,
  .6, 2, .2, 9, 8,
  .4, 1, 6, 5, 4,
  .2, 3, 10, 8, 2,
  .5, 2, 8, 6, 5,
  n, 0.2, 0.8,
  .6, 2, .2, 9, 8,
  .4, 1, 6, 5, 4,
  .2, 3, 10, 8, 2,
  .5, 2, 8, 6, 5,
  n, 0.9, 0.6,
  .6, 2, .2, 9, 8,
  .4, 1, 6, 5, 4,
  .2, 3, 10, 8, 2,
  .5, 2, 8, 6, 5,
  n, 0.3, 0.1,
  .6, 2, .2, 9, 8,
  .4, 1, 6, 5, 4,
  .2, 3, 10, 8, 2,
  .5, 2, 8, 6, 5,
  n, 0.8, 0.2,
  .6, 2, .2, 9, 8,
  .4, 1, 6, 5, 4,
  .2, 3, 10, 8, 2,
  .5, 2, 8, 6, 5,
  n, 0.6, 0.9,
  .8, 2, .2, 9, 8,
  .2, 1, 6, 5, 4,
  .5, 3, 10, 8, 2,
  .7, 2, 8, 6, 5,
  n, 0.1, 0.3,
  .8, 2, .2, 9, 8,
  .2, 1, 6, 5, 4,
  .5, 3, 10, 8, 2,
  .7, 2, 8, 6, 5,
  n, 0.2, 0.8,
  .8, 2, .2, 9, 8,
  .2, 1, 6, 5, 4,
  .5, 3, 10, 8, 2,
  .7, 2, 8, 6, 5,
  n, 0.9, 0.6,
  .8, 2, .2, 9, 8,
  .2, 1, 6, 5, 4,
  .5, 3, 10, 8, 2,
  .7, 2, 8, 6, 5,
  n, 0.3, 0.1,
  .8, 2, .2, 9, 8,
  .2, 1, 6, 5, 4,
  .5, 3, 10, 8, 2,
  .7, 2, 8, 6, 5,
  n, 0.8, 0.2,
  .8, 2, .2, 9, 8,
  .2, 1, 6, 5, 4,
  .5, 3, 10, 8, 2,
  .7, 2, 8, 6, 5,
  n, 0.6, 0.9,
  .8, 2, .2, 9, 8,
  .6, 1, 6, 3, .4,
  .5, 3, 10, .6, 2,
  .7, .2, 8, 6, .5,
  n, 0.1, 0.3,
  .8, 2, .2, 9, 8,
  .6, 1, 6, 3, .4,
  .5, 3, 10, .6, 2,
  .7, .2, 8, 6, .5,
  n, 0.2, 0.8,
  .8, 2, .2, 9, 8,
  .6, 1, 6, 3, .4,
  .5, 3, 10, .6, 2,
  .7, .2, 8, 6, .5,
  n, 0.9, 0.6,
  .8, 2, .2, 9, 8,
  .6, 1, 6, 3, .4,
  .5, 3, 10, .6, 2,
  .7, .2, 8, 6, .5,
  n, 0.3, 0.1,
  .8, 2, .2, 9, 8,
  .6, 1, 6, 3, .4,
  .5, 3, 10, .6, 2,
  .7, .2, 8, 6, .5,
  n, 0.8, 0.2,
  .8, 2, .2, 9, 8,
  .6, 1, 6, 3, .4,
  .5, 3, 10, .6, 2,
  .7, .2, 8, 6, .5,
  n, 0.6, 0.9,
  .8, 2, .2, 9, 8,
  .7, .2, 8, 6, .5,
  .6, 1, 6, 3, .4,
  .5, 3, 10, .6, 2,
  n, 0.1, 0.3,
  .8, 2, .2, 9, 8,
  .7, .2, 8, 6, .5,
  .6, 1, 6, 3, .4,
  .5, 3, 10, .6, 2,
  n, 0.2, 0.8,
  .8, 2, .2, 9, 8,
  .7, .2, 8, 6, .5,
  .6, 1, 6, 3, .4,
  .5, 3, 10, .6, 2,
  n, 0.9, 0.6,
  .8, 2, .2, 9, 8,
  .7, .2, 8, 6, .5,
  .6, 1, 6, 3, .4,
  .5, 3, 10, .6, 2,
  n, 0.3, 0.1,
  .8, 2, .2, 9, 8,
  .7, .2, 8, 6, .5,
  .6, 1, 6, 3, .4,
  .5, 3, 10, .6, 2,
  n, 0.8, 0.2,
  .8, 2, .2, 9, 8,
  .7, .2, 8, 6, .5,
  .6, 1, 6, 3, .4,
  .5, 3, 10, .6, 2,
)


df <- df |>
  rowwise() |>
  mutate(res = list(get_mae(
    n, theta_m, theta_h,
    cmw1, acm1, bcm1, acm2, bcm2,
    wmw1, awm1, bwm1, awm2, bwm2,
    chw1, ach1, bch1, ach2, bch2,
    whw1, awh1, bwh1, awh2, bwh2
  ))) |>
  unnest_wider(res) |>
  arrange(desc(mae)) |>
  glimpse()
