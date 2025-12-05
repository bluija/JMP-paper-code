library(assertthat)

d_to_auc <- function(d) {
  pnorm(d / sqrt(2))
}

auc_to_d <- function(auc) {
  qnorm(auc) * sqrt(2)
}

# 1) Combined accuracy based on Cohen's d
combined_accuracy_d <- function(theta_m, theta_h, d_m, d_h) {
  # Inputs:
  #  theta_m, theta_h : base accuracies for machine and human
  #  d_m, d_h         : Cohen's d for machine and human metacognitive sensitivity

  # Boundary cases for theta
  # Check for invalid theta values
  if (is.na(theta_m) || is.na(theta_h) || theta_m < 0 || theta_m > 1 || theta_h < 0 || theta_h > 1) {
    warning("theta values must be between 0 and 1")
    return(NA)
  }
  # If either accuracy is perfect, combined is perfect
  if (theta_m == 1 || theta_h == 1) {
    return(1)
  }
  # If either accuracy is zero, combined reduces to the other accuracy
  if (theta_m == 0) {
    return(theta_h)
  }
  if (theta_h == 0) {
    return(theta_m)
  }

  # Boundary cases for sensitivity d
  # No metacognitive sensitivity for both => no discrimination
  if (d_m == 0 && d_h == 0) {
    return(theta_m * theta_h)
  }
  # One infinite sensitivity => perfect discrimination for that observer
  if (is.infinite(d_m) && is.finite(d_h)) {
    # always trust machine when correct, else human
    return(theta_m + (1 - theta_m) * theta_h)
  }
  if (is.infinite(d_h) && is.finite(d_m)) {
    # always trust human when correct, else machine
    return(theta_h + (1 - theta_h) * theta_m)
  }
  # Both infinite sensitivity: perfect discrimination for both
  if (is.infinite(d_m) && is.infinite(d_h)) {
    # combined correct unless both wrong
    return(1 - (1 - theta_m) * (1 - theta_h))
  }

  # total squared d
  D2 <- d_m^2 + d_h^2

  # S = (logit(theta_m) - logit(theta_h)) / sqrt(D2)
  S <- (qlogis(theta_m) - qlogis(theta_h)) / sqrt(D2)

  # D = sqrt(D2) / 2
  D <- sqrt(D2) / 2

  # Combine
  p_xy1 <- theta_m * theta_h +
    theta_m * (1 - theta_h) * pnorm(S + D) +
    (1 - theta_m) * theta_h * (1 - pnorm(S - D))

  return(p_xy1)
}

# Examples
combined_accuracy_d(theta_m = 0.8, theta_h = 0.7, d_m = 1.2, d_h = 0.9)
