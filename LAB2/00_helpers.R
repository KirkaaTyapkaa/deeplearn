# Спільні статистичні функції для ЛР2
# Джерела: Лекція 2 (оцінки, SE), Лекція 3 (ДІ, тести Волда), Лекція 4 (бутстреп)


.libPaths(c("/home/nastyaj/R/custom-lib", .libPaths()))

required_packages <- c(
  "tidyverse", "boot"
)

installed <- rownames(installed.packages())
missing_packages <- required_packages[!required_packages %in% installed]

if (length(missing_packages) > 0) {
  install.packages(
    missing_packages,
    repos = "https://cloud.r-project.org"
  )
}

suppressPackageStartupMessages({
  library(tidyverse)
  library(boot)
})

# ---------- Група 1: асимптотично нормальні ДІ ----------

# CI для середнього: SE = s/sqrt(n), асимпт. N(0,1)
ci_mean <- function(x, alpha = 0.05) {
  x <- x[!is.na(x)]
  n <- length(x); m <- mean(x); se <- sd(x) / sqrt(n); z <- qnorm(1 - alpha/2)
  tibble(theta = m, se = se, lower = m - z*se, upper = m + z*se, n = n)
}

# CI для частки p-hat бінарної змінної: SE = sqrt(p*(1-p)/n)
ci_proportion <- function(x_binary, alpha = 0.05) {
  x <- as.integer(x_binary[!is.na(x_binary)])
  n <- length(x); p <- mean(x); se <- sqrt(p*(1-p)/n); z <- qnorm(1 - alpha/2)
  tibble(theta = p, se = se, lower = p - z*se, upper = p + z*se, n = n)
}

# CI для дисперсії: SE^2 = (mu4 - s^4)/n, де mu4 = E[(X-muX)^4] — Лекція 2
ci_variance <- function(x, alpha = 0.05) {
  x <- x[!is.na(x)]
  n <- length(x); s2 <- var(x); mu4 <- mean((x - mean(x))^4)
  se <- sqrt(pmax((mu4 - s2^2) / n, 0)); z <- qnorm(1 - alpha/2)
  tibble(theta = s2, se = se, lower = s2 - z*se, upper = s2 + z*se, n = n)
}

# CI для різниці середніх (Welch SE): SE = sqrt(s1^2/n1 + s2^2/n2)
ci_diff_means <- function(x, y, alpha = 0.05) {
  x <- x[!is.na(x)]; y <- y[!is.na(y)]
  n1 <- length(x); n2 <- length(y)
  d <- mean(x) - mean(y)
  se <- sqrt(var(x)/n1 + var(y)/n2); z <- qnorm(1 - alpha/2)
  tibble(theta = d, se = se, lower = d - z*se, upper = d + z*se, n1 = n1, n2 = n2)
}

# CI для різниці часток: SE = sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
ci_diff_proportions <- function(x1, x2, alpha = 0.05) {
  x1 <- as.integer(x1[!is.na(x1)]); x2 <- as.integer(x2[!is.na(x2)])
  n1 <- length(x1); n2 <- length(x2)
  p1 <- mean(x1); p2 <- mean(x2); d <- p1 - p2
  se <- sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2); z <- qnorm(1 - alpha/2)
  tibble(theta = d, se = se, lower = d - z*se, upper = d + z*se, n1 = n1, n2 = n2)
}

# ---------- Тест Волда: T = (theta_hat - theta0) / se ~ N(0,1) — Лекція 3 ----------

wald_test <- function(theta_hat, se, theta0 = 0, alternative = "two.sided",
                      hypothesis_id = "", description = "", h0 = "", h1 = "",
                      n_total = NA, related_question = "") {
  z <- (theta_hat - theta0) / se
  p <- switch(alternative,
              "two.sided" = 2 * (1 - pnorm(abs(z))),
              "less"      = pnorm(z),
              "greater"   = 1 - pnorm(z))
  tibble(
    hypothesis_id    = hypothesis_id,
    description      = description,
    h0               = h0,
    h1               = h1,
    alternative      = alternative,
    theta_hat        = theta_hat,
    se               = se,
    test_stat        = z,
    p_value          = p,
    n_total          = n_total,
    related_question = related_question
  )
}

# ---------- Група 2: бутстреп — Лекція 4 ----------

# Статистика для boot(): лише медіана (без inner bootstrap).
# Studentized CI вимагає inner bootstrap → O(B × R_inner × n log n) → зависає при n > 10k.
# Для Group 2 studentized є "бажано", не обов'язково → використовуємо 4 методи без нього.
boot_stat_median_simple <- function(x, indices) median(x[indices])

# Статистика для boot(): різниця медіан двох вибірок (strata-bootstrap)
boot_stat_median_diff <- function(d, indices) {
  db <- d[indices, ]
  median(db$v[db$g == 0]) - median(db$v[db$g == 1])
}

# Витяг 3-4 методів boot.ci (без studentized) у tidy tibble — для Group 2
# BCa може бути відсутнім (bca.ci fails for discrete stats like median)
extract_boot_ci_4 <- function(bci, statistic_name) {
  perc    <- bci$percent
  methods <- c("normal", "basic", "percentile")
  lower   <- c(bci$normal[2], bci$basic[4], perc[4])
  upper   <- c(bci$normal[3], bci$basic[5], perc[5])
  if (!is.null(bci$bca)) {
    methods <- c(methods, "bca")
    lower   <- c(lower,   bci$bca[4])
    upper   <- c(upper,   bci$bca[5])
  }
  tibble(statistic = statistic_name, estimate = bci$t0[1],
         method = methods, lower = lower, upper = upper)
}

# boot.ci з BCa fallback — BCa іноді не обчислюється для медіан (a = NA)
boot_ci_4 <- function(b, ...) {
  tryCatch(
    boot.ci(b, type = c("norm", "basic", "perc", "bca"), ...),
    error = function(e) {
      message("BCa недоступний (", conditionMessage(e), ") — 3 методи")
      boot.ci(b, type = c("norm", "basic", "perc"), ...)
    }
  )
}

# Витяг 5 методів boot.ci (включно зі studentized) — для Group 3 (кореляції, ОБОВ'ЯЗКОВО)
extract_boot_ci <- function(bci, statistic_name) {
  perc <- bci$percent %||% bci$perc
  tibble(
    statistic = statistic_name,
    estimate  = bci$t0[1],
    method    = c("normal", "basic", "studentized", "percentile", "bca"),
    lower     = c(bci$normal[2], bci$basic[4], bci$student[4], perc[4], bci$bca[4]),
    upper     = c(bci$normal[3], bci$basic[5], bci$student[5], perc[5], bci$bca[5])
  )
}

# Бутстреп для кореляції (Pearson або Spearman) + SE для studentized — Group 3
# Inner bootstrap на підвибірці n_inner щоб уникнути зависання
boot_stat_cor <- function(d, indices, method = "spearman", R_inner = 25, n_inner = 5000) {
  db <- d[indices, ]
  r  <- cor(db[[1]], db[[2]], method = method)
  # Підвибірка для inner bootstrap: не потрібна вся вибірка для оцінки дисперсії
  if (nrow(db) > n_inner) db_in <- db[sample(nrow(db), n_inner), ] else db_in <- db
  inner <- boot(db_in, R = R_inner, simple = TRUE,
                statistic = function(dd, ii) cor(dd[[1]][ii], dd[[2]][ii], method = method))
  c(r, var(inner$t))
}

# ---------- Завантаження даних (спільний шаблон) ----------

load_clean_data <- function() {
  candidates <- c(
    "clean_v2.3.rds",
    file.path(getwd(), "clean_v2.3.rds")
  )
  
  for (p in candidates) {
    if (file.exists(p)) {
      cat(sprintf("Завантаження: %s\n", normalizePath(p)))
      return(readRDS(p))
    }
  }
  
  stop("clean_v2.3.rds не знайдено. Запусти P4.R з папки LAB2 або вкажи шлях вручну.")
}

`%||%` <- function(a, b) if (!is.null(a)) a else b
