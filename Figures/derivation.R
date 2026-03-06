library(assertthat)

d_to_auc <- function(d) {
  pnorm(d / sqrt(2))
}

auc_to_d <- function(auc) {
  qnorm(auc) * sqrt(2)
}

combined_accuracy_d <- function(theta_m, theta_h, d_m, d_h, cov_hm = 0) {
  # Inputs:
  # theta_m, theta_h : base accuracies for machine and human
  # d_m, d_h         : Cohen's d for machine and human metacognitive sensitivities
  # cov_hm           : covariance between machine and human decisions (default 0)

  # Boundary cases for theta
  # Check for invalid theta values
  if (is.na(theta_m) ||
    is.na(theta_h) ||
    theta_m < 0 ||
    theta_m > 1 ||
    theta_h < 0 ||
    theta_h > 1) {
    warning("theta values must be between 0 and 1")
    return(NA)
  }
  # If either accuracy is 1, combined is also 1
  if (theta_m == 1 || theta_h == 1) {
    return(1)
  }
  # If either accuracy is 0, combined reduces to 2nd agent's accuracy
  if (theta_m == 0) {
    return(theta_h)
  }
  if (theta_h == 0) {
    return(theta_m)
  }

  # check covariance is within bounds given theta values
  rho <- cov_hm / sqrt(theta_m * (1 - theta_m) * theta_h * (1 - theta_h))
  print(rho)
  if (abs(rho) > 1) {
    warning("|rho| > 1, check cov_hm and theta values")
    return(NA)
  }

  # Boundary cases for sensitivity d
  # No metacognitive sensitivity for both => no discrimination
  if (d_m == 0 && d_h == 0) {
    return(max(theta_m, theta_h))
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

  # joint probabilities
  p11 <- theta_m * theta_h + cov_hm # both correct
  p10 <- theta_m * (1 - theta_h) - cov_hm # machine correct, human wrong
  p01 <- (1 - theta_m) * theta_h - cov_hm # machine wrong, human correct

  # Combine
  p_xy1 <- p11 +
    p10 * pnorm(S + D) +
    p01 * pnorm(D - S)

  return(p_xy1)
}

# Examples
combined_accuracy_d(theta_m = 0.8, theta_h = 0.8, d_m = 1.2, d_h = 0.9)
combined_accuracy_d(theta_m = 0.8, theta_h = 0.8, d_m = 1.2, d_h = 0.9, cov_hm = 0.1)
combined_accuracy_d(theta_m = 0.8, theta_h = 0.8, d_m = 1.2, d_h = 0.9, cov_hm = -0.1)

combined_accuracy_d(theta_m = 0.4, theta_h = 0.6, d_m = 1.2, d_h = 0.9)
combined_accuracy_d(theta_m = 0.4, theta_h = 0.6, d_m = 1.2, d_h = 0.9, cov_hm = 0.2)
combined_accuracy_d(theta_m = 0.4, theta_h = 0.6, d_m = 1.2, d_h = 0.9, cov_hm = -0.2)

combined_accuracy_d(theta_m = 0.2, theta_h = 0.8, d_m = 1.2, d_h = 0.9)
combined_accuracy_d(theta_m = 0.2, theta_h = 0.8, d_m = 1.2, d_h = 0.9, cov_hm = 0.1)
combined_accuracy_d(theta_m = 0.2, theta_h = 0.8, d_m = 1.2, d_h = 0.9, cov_hm = -0.1)
