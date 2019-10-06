
# Clear workspace
rm(list = ls())
# Set seed for reproducibility
set.seed(1234)
install.packages(c("aod", "car", "evtree", "fBasics", "kableExtra", "psych"))
install.packages("ggplot2")
install.packages("corrplot")
install.packages("grf")

library(ggplot2)
library(corrplot)
library(dplyr)       # Data manipulation (0.8.0.1)
library(fBasics)     # Summary statistics (3042.89)
library(corrplot)    # Correlations (0.84)
library(psych)       # Correlation p-values (1.8.12)
library(grf)         # Generalized random forests (0.10.2)
library(rpart)       # Classification and regression trees, or CART (4.1-13)
library(rpart.plot)  # Plotting tres (3.0.6)
library(treeClust)   # Predicting leaf position for causal trees (1.1-7)
library(car)         # linear hypothesis testing for causal tree (3.0-2)
library(devtools)    # Install packages from github (2.0.1)
library(readr)       # Reading csv files (1.3.1)
library(tidyr)       # Database operations (0.8.3)
library(tibble)      # Modern alternative to data frames (2.1.1)
library(knitr)       # RMarkdown (1.21)
library(kableExtra)  # Prettier RMarkdown (1.0.1)
library(ggplot2)     # general plotting tool (3.1.0)
library(haven)       # read stata files (2.0.0)
library(aod)         # hypothesis testing (1.3.1)
library(evtree)      # evolutionary learning of globally optimal trees (1.0-7)
library(foreign)



install_github('susanathey/causalTree') 
library(causalTree)


install_github("soerenkuenzel/causalToolbox")
install.packages("BART")
library(causalToolbox)

library(haven) 
df <- read_dta("C:/Users/Sakshi/Google Drive/ATLAS - Interns/Machine Learning/Data/ABA/ABA All Merged 20190920.dta")
View(df)


# Specify outcome, treatment, and covariate variable names to use
#covariate_names1- Hard, covariate_names2- Psych, covariate_names3- Hard+Psych
# Outcome- e_monthly_profits_all,e_monthly_revenue_all,e_monthly_expenses_all, e_wagebill_all, e_hh_consumption




outcome_variable_name <- "e_monthly_profits_all"
treatment_variable_name <- "treatment"

#Using all Baseline var as covariates. 

covariate_names1 = c("b_birth_year", 
                    "b_gender", 
                    "b_education", 
                    "b_income_sources", 
                    "b_other_businesses", 
                    "b_personal_expenses", 
                    "b_monthly_income", 
                    "b_sector_services", 
                    "b_sector_retail", 
                    "b_sector_manufacturing", 
                    "b_sector_agriculture", 
                    "b_sector_wholesale", 
                    "b_number_partners", 
                    "b_registration", 
                    "b_premises", 
                    "b_premisesown_own", 
                    "b_premisesown_rentold", 
                    "b_premisesown_rentnew", 
                    "b_start_year", 
                    "b_hours", 
                    "b_days", 
                    "b_branches", 
                    "b_closed_branches", 
                    "b_new_branches", 
                    "b_monthly_expenses", 
                    "b_monthly_revenue", 
                    "b_monthly_profits", 
                    "b_amount", 
                    "b_number_suppliers", 
                    "b_importing_inputs", 
                    "b_exporting", 
                    "b_sales_better", 
                    "b_sales_average", 
                    "b_sales_worse", 
                    "b_consignment", 
                    "b_credit", 
                    "b_staff_yesno", 
                    "b_num_employ_full", 
                    "b_num_employ_part ", 
                    "b_num_employ_temp ", 
                    "b_num_employ_intern", 
                    "b_num_employ_unpaid", 
                    "b_nemployees_main", 
                    "b_full_wage_main", 
                    "b_temp_wage_main", 
                    "b_intern_wage_main", 
                    "b_unpaid_wage_main", 
                    "b_wagebill_main", 
                    "digitspan", 
                    "b_separate_accounts", 
                    "b_fin_loan_reject_times", 
                    "b_fin_calcexp", 
                    "b_loans", 
                    "b_totloans_bank", 
                    "b_totloans_mfi", 
                    "b_totloans_famfriend", 
                    "b_totloans_rosca", 
                    "b_totloans", 
                    "raven_score", 
                    "fl_score", 
                    "b_hours", 
                    "b_days", 
                    "b_workhours_week", 
                    "b_hasbranch", 
                    "b_branches", 
                    "b_closed_branches", 
                    "b_new_branches", 
                    "b_monthly_expenses", 
                    "b_monthly_revenue", 
                    "b_monthly_profits", 
                    "b_selfpay_hourlywage", 
                    "b_selfpay_fixedsalary", 
                    "b_selfpay_commission", 
                    "b_selfpay_profitshare", 
                    "b_selfpay_expenses", 
                    "b_amount", 
                    "b_number_suppliers", 
                    "b_importing_inputs", 
                    "b_exporting", 
                    "b_sales_better", 
                    "b_sales_average", 
                    "b_sales_worse", 
                    "b_seasonal_sales", 
                    "b_lastmonth_sales", 
                    "b_beflastmonth_sales", 
                    "b_highest_profits", 
                    "b_lowest_profits", 
                    "b_risk_attitude", 
                    "b_hh_consumption")



# Combine all names
all_variables_names <- c(outcome_variable_name, treatment_variable_name, covariate_names1)
df <- df[, which(names(df) %in% all_variables_names)]
#1935 obs


# Drop rows containing missing values
df <- na.omit(df)

#1648 obs now





#Rename variables
names(df)[names(df) == outcome_variable_name] <- "Y"
names(df)[names(df) == treatment_variable_name] <- "W"

# Converting all columns to numerical
df <- data.frame(lapply(df, function(x) as.numeric(as.character(x))))

# Use train_fraction % of the dataset to train our models
train_fraction <- 0.80  
n <- dim(df)[1]
train_idx <- sample.int(n, replace=F, size=floor(n*train_fraction))
df_train <- df[train_idx,]
df_test <- df[-train_idx,]

summ_stats <- fBasics::basicStats(df)
summ_stats <- as.data.frame(t(summ_stats))

# Rename some of the columns for convenience
summ_stats <- summ_stats[c("Mean", "Stdev", "Minimum", "1. Quartile", "Median",  "3. Quartile", "Maximum")]
colnames(summ_stats)[colnames(summ_stats) %in% c('1. Quartile', '3. Quartile')] <- c('Lower quartile', 'Upper quartile')

summ_stats_table <- kable(summ_stats, "html", digits = 2)
kable_styling(summ_stats_table,
              bootstrap_options=c("striped", "hover", "condensed", "responsive"),
              full_width=FALSE)

pairwise_pvalues <- psych::corr.test(df, df)$p
corrplot(cor(df),
         type="upper",
         tl.col="black",
         order="hclust",
         tl.cex=1,
         addgrid.col = "black",
         p.mat=pairwise_pvalues,
         sig.level=0.05,
         number.font=10,
         insig="blank")

# __________________________________________________HTE 1: Causal Trees______________________________________________________________________________
#### Step 1: Split the dataset
# Diving the data 40%-40%-20% into splitting, estimation and validation samples
split_size <- floor(nrow(df_train) * 0.5)
split_idx <- sample(nrow(df_train), replace=FALSE, size=split_size)

# Make the splits
df_split <- df_train[split_idx,]
df_est <- df_train[-split_idx,]

#### Step 2: Fit the tree
fmla_ct <- paste("factor(Y) ~", paste(covariate_names, collapse = " + "))



ct_unpruned <- honest.causalTree(
  formula=fmla_ct,            # Define the model
  data=df_split,              # Subset used to create tree structure
  est_data=df_est,            # Which data set to use to estimate effects
  
  treatment=df_split$W,       # Splitting sample treatment variable
  est_treatment=df_est$W,     # Estimation sample treatment variable
  
  split.Rule="CT",            # Define the splitting option
  cv.option = "fit",         # Cross validation options. Sakshi Changed it from TOT to fit option
  
  split.Honest=TRUE,          # Use honesty when splitting
  cv.Honest=TRUE,             # Use honesty when performing cross-validation,
 
  minsize=25,                 # Min. number of treatment and control cases in each leaf
  HonestSampleSize=nrow(df_est)) # Num obs used in estimation after building the tree



#### Step 3: Cross-validate
# Table of cross-validated values by tuning parameter.
ct_cptable <- as.data.frame(ct_unpruned$cptable)

# Obtain optimal complexity parameter to prune tree.
selected_cp <- which.min(ct_cptable$xerror)
optim_cp_ct <- ct_cptable[selected_cp, "CP"]

# Prune the tree at optimal complexity parameter.
ct_pruned <- prune(tree=ct_unpruned, cp=optim_cp_ct)

#### Step 4: Predict point estimates (on estimation sample)
tauhat_ct_est <- predict(ct_pruned, newdata=df_est)

#### Step 5: Compute standard errors
# Create a factor column 'leaf' indicating leaf assignment
num_leaves <- length(unique(tauhat_ct_est))  # There are as many leaves as there are predictions
df_est$leaf <- factor(tauhat_ct_est, labels = seq(num_leaves))



# Run the regression
ols_ct <- lm(as.formula("Y ~ 0 + leaf + W:leaf"), data=df_est)
#Error , check var
str(df_est)
# may be check leaves now- We are only getting 1 leaf and I guess that is the problem. Let us reduce the ,minsize and see
#Using Atheys https://github.com/susanathey/causalTree/blob/master/briefintro.pdf 3.3 example, I changed tree estimation waya little bit. Changes are mentioned in CT_unpruned line
#Now I get 9 leaves


ols_ct_summary <- summary(ols_ct)
te_summary <- coef(ols_ct_summary)[(num_leaves+1):(2*num_leaves), c("Estimate", "Std. Error")]

kable_styling(kable(te_summary, "html", digits = 4, caption="Average treatment effects per leaf"),
              bootstrap_options=c("striped", "hover", "condensed", "responsive"),
              full_width=FALSE)

#### Step 6: Predict point estimates (on test set)
tauhat_ct_test <- predict(ct_pruned, newdata=df_test)


rpart.plot(
  x=ct_pruned,        # Pruned tree
  type=3,             # Draw separate split labels for the left and right directions
  fallen=TRUE,        # Position the leaf nodes at the bottom of the graph
  leaf.round=1,       # Rounding of the corners of the leaf node boxes
  extra=100,          # Display the percentage of observations in the node
  branch=.1,          # Shape of the branch lines
  box.palette="RdBu") # Palette for coloring the node

#### Treatment effect heterogeneity
# Null hypothesis: all leaf values are the same
hypothesis <- paste0("leaf1:W = leaf", seq(2, num_leaves), ":W")
ftest <- linearHypothesis(ols_ct, hypothesis, test="F")

kable_styling(kable(data.frame(ftest, check.names = FALSE, row.names = NULL)[2,],
                    "html", digits = 4,
                    caption="Testing null hypothesis:<br> Average treatment effect is same across leaves"),
              bootstrap_options=c("striped", "hover", "condensed", "responsive"),
              full_width=FALSE)

# Null hypothesis: leaf i = leaf k for all i != k
p_values_leaf_by_leaf <- matrix(nrow = num_leaves, ncol = num_leaves)
differences_leaf_by_leaf <- matrix(nrow = num_leaves, ncol = num_leaves)
stderror_leaf_by_leaf <- matrix(nrow = num_leaves, ncol = num_leaves)
hypotheses_grid <- combn(1:num_leaves, 2)
summ <- coef(summary(ols_ct))

invisible(apply(hypotheses_grid, 2, function(x) {
  leafi <- paste0("leaf", x[1], ":W")
  leafj <- paste0("leaf", x[2], ":W")
  hypothesis <- paste0(leafi, " = ", leafj)
  
  differences_leaf_by_leaf[x[2], x[1]] <<- summ[leafj, 1] - summ[leafi, 1]
  stderror_leaf_by_leaf[x[2], x[1]] <<- sqrt(summ[leafj, 2]^2 + summ[leafi, 2]^2)
  p_values_leaf_by_leaf[x[2], x[1]] <<- linearHypothesis(ols_ct, hypothesis)[2, "Pr(>F)"]
}))

# Little trick to display p-values under mean difference values in HTML
diffs <- matrix(nrow = num_leaves, ncol = num_leaves)
invisible(apply(hypotheses_grid, 2, function(x) {
  d <- differences_leaf_by_leaf[x[2], x[1]]
  s <- stderror_leaf_by_leaf[x[2], x[1]]
  p <- p_values_leaf_by_leaf[x[2], x[1]]
  top <- cell_spec(round(d, 3), "html",
                   background=ifelse(is.na(p) || (p > 0.1), "white", "gray"),
                   color=ifelse(is.na(p), "white", ifelse(p < 0.1, "white", "gray")))
  value <- ifelse(is.na(p), "", paste0(top, " <br> ", "(", round(s, 3), ")"))
  diffs[x[2], x[1]] <<- value
}))

diffs <- as.data.frame(diffs) %>% mutate_all(as.character)
rownames(diffs) <- paste0("leaf", 1:num_leaves)
colnames(diffs) <- paste0("leaf", 1:num_leaves)

# Title of table
caption <- "Pairwise leaf differences:<br>
Average treatment effect differences between leaf i and leaf j"

# Styling color and background
color <-function(x) ifelse(is.na(x), "white", "gray")

diffs %>%
  rownames_to_column() %>%
  mutate_all(function(x) cell_spec(x, "html", escape=FALSE, color=color(x))) %>%
  kable(format="html", caption=caption, escape = FALSE) %>%
  kable_styling(bootstrap_options=c("condensed", "responsive"), full_width=FALSE) %>%
  footnote(general='Standard errors in parenthesis. Significance (not adjusted for multiple testing):
           <ul>
           <li>No background color: p ≥ 0.1
           <li><span style="color: white;border-radius: 4px; padding-right: 4px; padding-left: 4px; background-color: gray;">Gray</span> background: p < 0.1
           <li><span style="color: white;border-radius: 4px; padding-right: 4px; padding-left: 4px; background-color: black;">Black</span> background: p < 0.05
           </ul>
           ', escape=F)

# Null hypothesis: the mean is equal across all leaves
hypothesis <- paste0("leaf1 = leaf", seq(2, num_leaves))
means_per_leaf <- matrix(nrow = num_leaves, ncol = num_leaves)
significance <- matrix(nrow = 2, ncol=length(covariate_names))

# Regress each covariate on leaf assignment to means p
cov_means <- lapply(covariate_names, function(covariate) {
  lm(paste0(covariate, ' ~ 0 + leaf'), data = df_est)
})

# Extract the mean and standard deviation of each covariate per leaf
cov_table <- lapply(cov_means, function(cov_mean) {
  as.data.frame(t(coef(summary(cov_mean))[,c("Estimate", "Std. Error")]))
})

# Test if means are the same across leaves
cov_ftests <- sapply(cov_means, function(cov_mean) {
  # Sometimes the regression has no residual (SSE = 0), 
  # so we cannot perform an F-test
  tryCatch({
    linearHypothesis(cov_mean, hypothesis)[2, c("F", "Pr(>F)")]
  },
  error = function(cond) {
    message(paste0("Error message during F-test for`", cov_mean$terms[[2]], "`:"))
    message(cond)
    return(c("F" = NA, "Pr(>F)" = NA))
  })
})

# Little trick to display the standard errors
table <- lapply(seq_along(covariate_names), function(j) {
  m <- round(signif(cov_table[[j]], digits=4), 3)
  m["Estimate",] <- as.character(m["Estimate",])
  m["Std. Error",] <- paste0("(", m["Std. Error",], ")")
  m
})
table <- do.call(rbind, table)

# Covariate names
covnames <- rep("", nrow(table))
covnames[seq(1, length(covnames), 2)] <-
  cell_spec(covariate_names, format = "html", escape = F, color = "black", bold = T)

table <- cbind(covariates=covnames, table)

# Title of table
caption <- "Average covariate values in each leaf"

table %>%
  kable(format="html", digits=2, caption=caption, escape = FALSE, row.names = FALSE) %>%
  kable_styling(bootstrap_options=c("condensed", "responsive"), full_width=FALSE)


# Adding a factor column turns all columns into character
df_est <- as.data.frame(apply(df_est, 2, as.numeric))

covariate_means_per_leaf <- aggregate(. ~ leaf, df_est, mean)[,covariate_names]
covariate_means <- apply(df_est, 2, mean)[covariate_names]
leaf_weights <- table(df_est$leaf) / dim(df_est)[1] 
deviations <- t(apply(covariate_means_per_leaf, 1, function(x) x - covariate_means))
covariate_means_weighted_var <- apply(deviations, 2, function(x) sum(leaf_weights * x^2))
covariate_var <- apply(df_est, 2, var)[covariate_names]
cov_variation <- covariate_means_weighted_var / covariate_var



sorted_cov_variation <- sort(cov_variation, decreasing = TRUE)
table <- as.data.frame(sorted_cov_variation)
colnames(table) <- NULL

kable_styling(kable(table,  "html", digits = 4, row.names=TRUE,
                    caption="Covariate variation across leaves"),
              bootstrap_options=c("striped", "hover", "condensed", "responsive"),
              full_width=FALSE)




##___________________________________________ HTE 2: Causal Forests and the R-Learner_______________

#### Step 1: Fit the forest
cf <- causal_forest(
  X = as.matrix(df_train[,covariate_names]),
  Y = df_train$Y,
  W = df_train$W,
  num.trees=200) # This is just for speed. In a real application, remember increase this number!

#### Step 2(a): Predict point estimates and standard errors (training set, out-of-bag)
oob_pred <- predict(cf, estimate.variance=TRUE)

kable_styling(kable(head(oob_pred, 3), "html", digits = 4),
              bootstrap_options=c("striped", "hover", "condensed", "responsive"),
              full_width=FALSE)
oob_tauhat_cf <- oob_pred$predictions
oob_tauhat_cf_se <- sqrt(oob_pred$variance.estimates)
test_pred <- predict(cf, newdata=as.matrix(df_test[covariate_names]), estimate.variance=TRUE)
tauhat_cf_test <- test_pred$predictions
tauhat_cf_test_se <- sqrt(test_pred$variance.estimates)
kable_styling(kable(head(test_pred, 3), "html", digits = 4),
              bootstrap_options=c("striped", "hover", "condensed", "responsive"),
              full_width=FALSE)

cf_known_prop <- grf::causal_forest(
  X = as.matrix(df_train[covariate_names]),
  Y = df_train$Y,
  W = df_train$W,
  W.hat = rep(mean(df_train$W), times=nrow(df_train)))  # Passing the known (approximate) propensity score

### Assessing heterogeneity

hist(oob_tauhat_cf, main="Causal forests: out-of-bag CATE")

var_imp <- c(variable_importance(cf))
names(var_imp) <- covariate_names
sorted_var_imp <- sort(var_imp, decreasing=TRUE)


as.data.frame(sorted_var_imp, row.names = names(sorted_var_imp)) %>%
  kable("html", digits = 4, row.names = T) %>%
  kable_styling(bootstrap_options=c("striped", "hover", "condensed", "responsive"), full_width=FALSE)


#### Heterogeneity across subgroups

# Manually creating subgroups
num_tiles <- 4  # ntiles = CATE is above / below the median
df_train$cate <- oob_tauhat_cf
df_train$ntile <- factor(ntile(oob_tauhat_cf, n=num_tiles))

##### Average treatment effects within subgroups
ols_sample_ate <- lm("Y ~ ntile + ntile:W", data=df_train)
estimated_sample_ate <- coef(summary(ols_sample_ate))[(num_tiles+1):(2*num_tiles), c("Estimate", "Std. Error")]
hypothesis_sample_ate <- paste0("ntile1:W = ", paste0("ntile", seq(2, num_tiles), ":W"))
ftest_pvalue_sample_ate <- linearHypothesis(ols_sample_ate, hypothesis_sample_ate)[2,"Pr(>F)"]

estimated_aipw_ate <- lapply(
  seq(num_tiles), function(w) {
    ate <- average_treatment_effect(cf, subset = df_train$ntile == w)
  })
estimated_aipw_ate <- data.frame(do.call(rbind, estimated_aipw_ate))

# Testing for equality using Wald test
waldtest_pvalue_aipw_ate <- wald.test(Sigma = diag(estimated_aipw_ate$std.err^2),
                                      b = estimated_aipw_ate$estimate,
                                      Terms = 1:num_tiles)$result$chi2[3]


# Round the estimates and standard errors before displaying them
estimated_sample_ate_rounded <- round(signif(estimated_sample_ate, digits = 5), 4)
estimated_aipw_ate_rounded <- round(signif(estimated_aipw_ate, digits = 5), 4)

# Format Table: Parenthesis, row/column names
sample_ate_w_se <- c(rbind(estimated_sample_ate[,"Estimate"], paste0("(", estimated_sample_ate_rounded[,"Std. Error"], ")")))
aipw_ate_w_se <- c(rbind(estimated_aipw_ate[,"estimate"], paste0("(", estimated_aipw_ate_rounded[,"std.err"], ")")))
table <- cbind("Sample ATE" = sample_ate_w_se, "AIPW ATE" = aipw_ate_w_se)
table <- rbind(table, round(signif(c(ftest_pvalue_sample_ate, waldtest_pvalue_aipw_ate), digits = 5), 4)) # add p-value to table
left_column <- rep('', nrow(table))
left_column[seq(1, nrow(table), 2)] <-
  cell_spec(c(paste0("ntile", seq(num_tiles)), "P-Value"),
            format = "html", escape = FALSE, color = "black", bold = TRUE)
table <- cbind(" " = left_column, table)

# Output table
table %>%
  kable("html", escape = FALSE, row.names = FALSE) %>%
  kable_styling(bootstrap_options=c("striped", "hover", "condensed", "responsive"), full_width=FALSE) %>%
  footnote(general = "Average treatment effects per subgroup defined by out-of-bag CATE.<br>
           P-value is testing <i>H<sub>0</sub>: ATE is constant across ntiles</i>.<br>
           Sample ATE uses an F-test and AIPW uses a Wald test;<br>
           see the code above for more details.",
           escape=FALSE)

# Transform to data tables with relevant columns
estimated_sample_ate <- as.data.frame(estimated_sample_ate)
estimated_sample_ate$Method <- "Sample ATE"
estimated_sample_ate$Ntile <- as.numeric(sub(".*([0-9]+).*", "\\1", rownames(estimated_sample_ate)))

estimated_aipw_ate <- as.data.frame(estimated_aipw_ate)
estimated_aipw_ate$Method <- "AIPW ATE"
estimated_aipw_ate$Ntile <- as.numeric(rownames(estimated_aipw_ate))

# unify column names and combine
colnames(estimated_sample_ate) <- c("Estimate", "SE", "Method", "Ntile")
colnames(estimated_aipw_ate) <- c("Estimate", "SE", "Method", "Ntile")
combined_ate_estimates <- rbind(estimated_sample_ate, estimated_aipw_ate)

# plot
ggplot(combined_ate_estimates) +
  geom_pointrange(aes(x = Ntile, y = Estimate, ymax = Estimate + 1.96 * SE, ymin = Estimate - 1.96 * SE, color = Method), 
                  size = 0.5,
                  position = position_dodge(width = .5)) +
  geom_errorbar(aes(x = Ntile, ymax = Estimate + 1.96 * SE, ymin = Estimate - 1.96 * SE, color = Method), 
                width = 0.4,
                size = 0.75,
                position = position_dodge(width = .5)) +
  theme_linedraw() +
  labs(x = "N-tile", y = "ATE Estimate", title = "ATE within N-tiles (as defined by predicted CATE)")


#### Heterogeneity across covariates
# Regress each covariate on ntile assignment to means p
cov_means <- lapply(covariate_names, function(covariate) {
  lm(paste0(covariate, ' ~ 0 + ntile'), data = df_train)
})

# Extract the mean and standard deviation of each covariate per ntile
cov_table <- lapply(cov_means, function(cov_mean) {
  as.data.frame(t(coef(summary(cov_mean))[,c("Estimate", "Std. Error")]))
})

# Little trick to display the standard errors
table <- lapply(seq_along(covariate_names), function(j) {
  m <- round(signif(cov_table[[j]], digits=4), 3)
  m["Estimate",] <- as.character(m["Estimate",])
  m["Std. Error",] <- paste0("(", m["Std. Error",], ")")
  m
})
table <- do.call(rbind, table)

# Covariate names
covnames <- rep("", nrow(table))
covnames[seq(1, length(covnames), 2)] <-
  cell_spec(covariate_names, format = "html", escape = F, color = "black", bold = T)

table <- cbind(covariates=covnames, table)

# Title of table
caption <- "Average covariate values in each n-tile"

table %>%
  kable(format="html", digits=2, caption=caption, escape = FALSE, row.names = FALSE) %>%
  kable_styling(bootstrap_options=c("condensed", "responsive"), full_width=FALSE)

covariate_means_per_ntile <- aggregate(. ~ ntile, df_train, mean)[,covariate_names]
covariate_means <- apply(df_train, 2, mean)[covariate_names]
ntile_weights <- table(df_train$ntile) / dim(df_train)[1] 
deviations <- t(apply(covariate_means_per_ntile, 1, function(x) x - covariate_means))
covariate_means_weighted_var <- apply(deviations, 2, function(x) sum(ntile_weights * x^2))
covariate_var <- apply(df_train, 2, var)[covariate_names]
cov_variation <- covariate_means_weighted_var / covariate_var

sorted_cov_variation <- sort(cov_variation, decreasing = TRUE)
table <- as.data.frame(sorted_cov_variation)
colnames(table) <- NULL

kable_styling(kable(table,  "html", digits = 4, row.names=TRUE,
                    caption="Covariate variation across n-tiles"),
              bootstrap_options=c("striped", "hover", "condensed", "responsive"),
              full_width=FALSE)


#### Partial dependence plots

if (dataset_name == "welfare") {
  var_of_interest = "polviews"
  vars_of_interest = c("income", "polviews")
} else {
  # Selecting a continuous variable, if available, to make for a more interesting graph
  continuous_variables <- sapply(covariate_names, function(x) length(unique(df_train[, x])) > 5)
  
  # Select variable for single variable plot
  var_of_interest <- ifelse(sum(continuous_variables) > 0,
                            covariate_names[continuous_variables][1],
                            covariate_names[1])
  
  # Select variables for two variable plot
  vars_of_interest <- c(var_of_interest,
                        ifelse(sum(continuous_variables) > 1,
                               covariate_names[continuous_variables][2],
                               covariate_names[covariate_names != var_of_interest][1]))
}

# Create a grid of values: if continuous, quantiles; else, plot the actual values
is_continuous <- (length(unique(df_train[,var_of_interest])) > 5) # crude rule for determining continuity
if(is_continuous) {
  x_grid <- quantile(df_train[,var_of_interest], probs = seq(0, 1, length.out = 5))
} else {
  x_grid <- sort(unique(df_train[,var_of_interest]))
}
df_grid <- setNames(data.frame(x_grid), var_of_interest)

# For the other variables, keep them at their median
other_covariates <- covariate_names[which(covariate_names != var_of_interest)]
df_median <- data.frame(lapply(df_train[,other_covariates], median))
df_eval <- crossing(df_median, df_grid)

# Predict the treatment effect
pred <- predict(cf, newdata=df_eval[,covariate_names], estimate.variance=TRUE)
df_eval$tauhat <- pred$predictions
df_eval$se <- sqrt(pred$variance.estimates)


# Change to factor so the plotted values are evenly spaced
df_eval[, var_of_interest] <- as.factor(round(df_eval[, var_of_interest], digits = 4))

# Descriptive labeling
label_description <- ifelse(is_continuous, '\n(Evaluated at quintiles)', '')

# Plot
df_eval %>%
  mutate(ymin_val = tauhat-1.96*se) %>%
  mutate(ymax_val = tauhat+1.96*se) %>%
  ggplot() +
  geom_line(aes_string(x=var_of_interest, y="tauhat", group = 1), color="red") +
  geom_errorbar(aes_string(x=var_of_interest,ymin="ymin_val", ymax="ymax_val", width=.2),color="blue") +
  xlab(paste0("Effect of ", var_of_interest, label_description)) +
  ylab("Predicted Treatment Effect") +
  theme_linedraw() +
  theme(axis.ticks = element_blank())

# Split up continuous and binary variables
binary_covariates <- sapply(covariate_names,
                            function(x) length(unique(df_train[, x])) <= 2)

evaluate_partial_dependency <- function(var_of_interest, is_binary) {
  if(is_binary){
    # Get two unique values for the variable
    x_grid <- sort(unique(df_train[,var_of_interest]))
  } else {
    # Get quartile values for the variable
    x_grid <- quantile(df_train[,var_of_interest], probs = seq(0, 1, length.out = 5))
  }
  df_grid <- setNames(data.frame(x_grid), var_of_interest)
  
  # For the other variables, keep them at their median
  other_covariates <- covariate_names[which(covariate_names != var_of_interest)]
  df_median <- data.frame(lapply(df_train[,other_covariates], median))
  df_eval <- crossing(df_median, df_grid)
  
  # Predict the treatment effect
  pred <- predict(cf, newdata=df_eval[,covariate_names], estimate.variance=TRUE)
  rbind('Tau Hat' = pred$predictions,
        'Std. Error' = sqrt(pred$variance.estimates))
}

# Make the table for non-binary variables
nonbinary_partial_dependency_tauhats <- lapply(covariate_names[!binary_covariates],
                                               function(variable) evaluate_partial_dependency(variable, FALSE))

# Make the table for binary variables
binary_partial_dependency_tauhats <- lapply(covariate_names[binary_covariates],
                                            function(variable) evaluate_partial_dependency(variable, TRUE))


if(sum(!binary_covariates) > 0) {
  # Little trick to display the standard errors
  table <- lapply(seq_along(covariate_names[!binary_covariates]), function(j) {
    m <- round(signif(nonbinary_partial_dependency_tauhats[[j]], digits=4), 3)
    m["Tau Hat",] <- as.character(m["Tau Hat",])
    m["Std. Error",] <- paste0("(", m["Std. Error",], ")")
    m
  })
  table <- do.call(rbind, table)
  colnames(table) <- paste0('Q=', names(quantile(NULL, probs = seq(0, 1, length.out = 5))))
  
  # Covariate names
  covnames <- rep("", nrow(table))
  covnames[seq(1, length(covnames), 2)] <-
    cell_spec(covariate_names[!binary_covariates], format = "html", escape = F, color = "black", bold = T)
  
  table <- cbind(covariates=covnames, table)
  
  # Title of table
  caption <- "The CATE function's value varying each non-binary covariate across its indicated percentile values, holding all other covariates at their medians."
  
  table %>%
    kable(format="html", digits=2, caption=caption, escape = FALSE, row.names = FALSE) %>%
    kable_styling(bootstrap_options=c("condensed", "responsive"), full_width=FALSE)
}


if(sum(binary_covariates) > 0) {
  # Little trick to display the standard errors
  table <- lapply(seq_along(covariate_names[binary_covariates]), function(j) {
    m <- round(signif(binary_partial_dependency_tauhats[[j]], digits=4), 3)
    m["Tau Hat",] <- as.character(m["Tau Hat",])
    m["Std. Error",] <- paste0("(", m["Std. Error",], ")")
    m
  })
  table <- do.call(rbind, table)
  colnames(table) <- paste0('X=',c('0', '1'))
  
  # Covariate names
  covnames <- rep("", nrow(table))
  covnames[seq(1, length(covnames), 2)] <-
    cell_spec(covariate_names[binary_covariates], format = "html", escape = F, color = "black", bold = T)
  
  table <- cbind(covariates=covnames, table)
  
  # Title of table
  caption <- "The CATE function's value varying each binary covariate at 0 and 1, holding all other covariates at their medians."
  
  table %>%
    kable(format="html", digits=2, caption=caption, escape = FALSE, row.names = FALSE) %>%
    kable_styling(bootstrap_options=c("condensed", "responsive"), full_width=FALSE)
}











