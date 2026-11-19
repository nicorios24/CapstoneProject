# ==========================================
# TE PREDICTIONS MODEL
# ==========================================

# Load libraries
library(tidyverse)
library(nflreadr)
library(torch)
library(luz)
library(ggplot2)
library(nflplotR)
library(slider)
library(recipes)
library(yardstick)

# ==========================================
# LOAD TE DATA
# ==========================================

te_data <- load_player_stats(seasons = 2020:2026, summary_level = "week") %>%
  filter(position == "TE")

dim(te_data)

glimpse(te_data)

te_data %>%
  count(season)

te_data %>%
  count(position)

te_model_data <- te_data %>%
  arrange(player_id, season, week) %>%
  group_by(player_id) %>%
  mutate(
    # Previous game statistics
    prev_receiving_yards = lag(receiving_yards),
    prev_receiving_tds = lag(receiving_tds),
    prev_targets = lag(targets),
    prev_receptions = lag(receptions),
    prev_rushing_yards = lag(rushing_yards),
    prev_carries = lag(carries),

    # Rolling averages: previous 3 games
    avg_receiving_yards_3 = slider::slide_dbl(
      lag(receiving_yards), mean, .before = 2, .complete = FALSE
    ),
    avg_targets_3 = slider::slide_dbl(
      lag(targets), mean, .before = 2, .complete = FALSE
    ),
    avg_receptions_3 = slider::slide_dbl(
      lag(receptions), mean, .before = 2, .complete = FALSE
    ),

    # Rolling averages: previous 5 games
    avg_receiving_yards_5 = slider::slide_dbl(
      lag(receiving_yards), mean, .before = 4, .complete = FALSE
    ),
    avg_targets_5 = slider::slide_dbl(
      lag(targets), mean, .before = 4, .complete = FALSE
    ),
    avg_receptions_5 = slider::slide_dbl(
      lag(receptions), mean, .before = 4, .complete = FALSE
    ),

    # Touchdown history
    prev_td_indicator = lag(as.integer(receiving_tds > 0), default = 0),

    td_rate_last_3 = slider::slide_dbl(
      lag(as.integer(receiving_tds > 0)), mean,
      .before = 2, .complete = FALSE
    ),

    td_rate_last_5 = slider::slide_dbl(
      lag(as.integer(receiving_tds > 0)), mean,
      .before = 4, .complete = FALSE
    )
  ) %>%
  ungroup()

dim(te_model_data)

te_model_data %>%
  select(
    player_display_name, season, week,
    prev_receiving_yards, prev_targets, prev_receptions,
    avg_receiving_yards_3, avg_receiving_yards_5,
    prev_td_indicator, td_rate_last_3, td_rate_last_5
  ) %>%
  head(15)

colSums(is.na(te_model_data %>%
  select(starts_with("prev_"), starts_with("avg_"), starts_with("td_"))))

te_team_stats <- load_team_stats(
  seasons = 2020:2026,
  summary_level = "week"
) %>%
  filter(season_type == "REG")

te_defense_data <- te_team_stats %>%
  filter(season_type == "REG") %>%
  transmute(
    season,
    week,
    game_id,
    defense_team = opponent_team,
    pass_yards_allowed = passing_yards,
    pass_tds_allowed = passing_tds,
    pass_attempts_allowed = attempts,
    pass_completions_allowed = completions,
    pass_epa_allowed = passing_epa
  )

dim(te_team_stats)
dim(te_defense_data)
head(te_defense_data)

te_defense_history <- te_defense_data %>%
  group_by(defense_team) %>%
  arrange(season, week, .by_group = TRUE) %>%
  mutate(
    # Previous game defensive statistics
    prev_pass_yards_allowed = lag(pass_yards_allowed),
    prev_pass_tds_allowed = lag(pass_tds_allowed),
    prev_pass_attempts_allowed = lag(pass_attempts_allowed),
    prev_pass_completions_allowed = lag(pass_completions_allowed),
    prev_pass_epa_allowed = lag(pass_epa_allowed),

    # Previous 3-game averages
    avg_pass_yards_allowed_3 = slider::slide_dbl(
      lag(pass_yards_allowed), mean, .before = 2, .complete = FALSE
    ),
    avg_pass_tds_allowed_3 = slider::slide_dbl(
      lag(pass_tds_allowed), mean, .before = 2, .complete = FALSE
    ),
    avg_pass_attempts_allowed_3 = slider::slide_dbl(
      lag(pass_attempts_allowed), mean, .before = 2, .complete = FALSE
    ),
    avg_pass_completions_allowed_3 = slider::slide_dbl(
      lag(pass_completions_allowed), mean, .before = 2, .complete = FALSE
    ),
    avg_pass_epa_allowed_3 = slider::slide_dbl(
      lag(pass_epa_allowed), mean, .before = 2, .complete = FALSE
    ),

    # Previous 5-game averages
    avg_pass_yards_allowed_5 = slider::slide_dbl(
      lag(pass_yards_allowed), mean, .before = 4, .complete = FALSE
    ),
    avg_pass_tds_allowed_5 = slider::slide_dbl(
      lag(pass_tds_allowed), mean, .before = 4, .complete = FALSE
    ),
    avg_pass_attempts_allowed_5 = slider::slide_dbl(
      lag(pass_attempts_allowed), mean, .before = 4, .complete = FALSE
    ),
    avg_pass_completions_allowed_5 = slider::slide_dbl(
      lag(pass_completions_allowed), mean, .before = 4, .complete = FALSE
    ),
    avg_pass_epa_allowed_5 = slider::slide_dbl(
      lag(pass_epa_allowed), mean, .before = 4, .complete = FALSE
    )
  ) %>%
  ungroup()

dim(te_defense_history)

te_defense_history %>%
  select(
    defense_team, season, week,
    prev_pass_yards_allowed,
    avg_pass_yards_allowed_3,
    avg_pass_yards_allowed_5
  ) %>%
  head(15)

colSums(is.na(te_defense_history %>%
  select(starts_with("prev_"), starts_with("avg_"))))

te_model_data <- te_model_data %>%
  left_join(
    te_defense_history %>%
      select(
        defense_team,
        season,
        week,
        starts_with("prev_pass_"),
        starts_with("avg_pass_")
      ),
    by = c(
      "opponent_team" = "defense_team",
      "season",
      "week"
    )
  )

dim(te_model_data)

te_model_data %>%
  select(
    player_display_name,
    team,
    opponent_team,
    season,
    week,
    prev_pass_yards_allowed,
    avg_pass_yards_allowed_3,
    avg_pass_yards_allowed_5
  ) %>%
  head(15)

colSums(is.na(te_model_data %>%
  select(starts_with("prev_pass_"), starts_with("avg_pass_"))))

te_data %>%
  count(season_type)

te_model_data %>%
  filter(is.na(prev_pass_yards_allowed)) %>%
  count(season_type)

te_model_data %>%
  filter(is.na(prev_pass_yards_allowed)) %>%
  count(season, season_type, week)

te_model_data <- te_model_data %>%
  filter(season_type == "REG")

dim(te_model_data)

te_model_data %>%
  count(season)

te_model_data %>%
  count(season_type)

colSums(is.na(te_model_data %>%
  select(starts_with("prev_pass_"), starts_with("avg_pass_"))))

# Training data: 2020-2024
te_train_data <- te_model_data %>%
  filter(season >= 2020, season <= 2024)

# Validation data: 2025
te_validation_data <- te_model_data %>%
  filter(season == 2025)

# Testing data: 2026
te_2026_data <- te_model_data %>%
  filter(season == 2026)

dim(te_train_data)
dim(te_validation_data)
dim(te_2026_data)

te_train_data %>% count(season)

te_validation_data %>% count(season)

te_2026_data %>% count(season)

# Create the receiving touchdown indicator
te_train_data <- te_train_data %>%
  mutate(any_receiving_td = as.integer(receiving_tds > 0))

te_validation_data <- te_validation_data %>%
  mutate(any_receiving_td = as.integer(receiving_tds > 0))

te_2026_data <- te_2026_data %>%
  mutate(any_receiving_td = as.integer(receiving_tds > 0))

te_train_targets <- te_train_data %>%
  select(targets, receptions, receiving_yards, any_receiving_td)

te_validation_targets <- te_validation_data %>%
  select(targets, receptions, receiving_yards, any_receiving_td)

te_2026_targets <- te_2026_data %>%
  select(targets, receptions, receiving_yards, any_receiving_td)

dim(te_train_targets)
dim(te_validation_targets)
dim(te_2026_targets)

te_train_targets %>%
  count(any_receiving_td)

te_validation_targets %>%
  count(any_receiving_td)

te_2026_targets %>%
  count(any_receiving_td)

# Historical TE player features
te_player_predictors <- grep(
  "^(prev_|avg_|td_)",
  names(te_train_data),
  value = TRUE
)

# NGS features
te_ngs_predictors <- grep(
  "^ngs_",
  names(te_train_data),
  value = TRUE
)

# Historical opponent defensive features
te_defense_predictors <- grep(
  "^(prev_pass_|avg_pass_)",
  names(te_train_data),
  value = TRUE
)

# Combine predictors
te_predictors <- unique(c(
  te_player_predictors,
  te_ngs_predictors,
  te_defense_predictors
))

length(te_predictors)

te_predictors

setdiff(te_predictors, names(te_train_data))

te_recipe <- recipe(
  ~ .,
  data = te_train_data %>%
    select(all_of(te_predictors))
) %>%
  step_indicate_na(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

te_prep <- prep(
  te_recipe,
  training = te_train_data %>%
    select(all_of(te_predictors))
)

te_train_processed <- bake(
  te_prep,
  new_data = te_train_data %>%
    select(all_of(te_predictors))
)

te_validation_processed <- bake(
  te_prep,
  new_data = te_validation_data %>%
    select(all_of(te_predictors))
)

te_2026_processed <- bake(
  te_prep,
  new_data = te_2026_data %>%
    select(all_of(te_predictors))
)

dim(te_train_processed)
dim(te_validation_processed)
dim(te_2026_processed)

sum(is.na(te_train_processed))
sum(is.na(te_validation_processed))
sum(is.na(te_2026_processed))

nrow(te_2026_data)

nrow(te_2026_processed)

nrow(te_2026_targets)

te_2026_data %>%
  count(season)

te_2026_data %>%
  count(player_id, game_id) %>%
  filter(n > 1)

# Calculate training target means
te_target_means <- te_train_targets %>%
  summarise(
    targets_mean = mean(targets),
    receptions_mean = mean(receptions),
    receiving_yards_mean = mean(receiving_yards)
  )

# Calculate training target standard deviations
te_target_sds <- te_train_targets %>%
  summarise(
    targets_sd = sd(targets),
    receptions_sd = sd(receptions),
    receiving_yards_sd = sd(receiving_yards)
  )

te_train_targets_scaled <- te_train_targets %>%
  mutate(
    targets = (targets - te_target_means$targets_mean) /
      te_target_sds$targets_sd,
    receptions = (receptions - te_target_means$receptions_mean) /
      te_target_sds$receptions_sd,
    receiving_yards = (receiving_yards - te_target_means$receiving_yards_mean) /
      te_target_sds$receiving_yards_sd
  )

te_validation_targets_scaled <- te_validation_targets %>%
  mutate(
    targets = (targets - te_target_means$targets_mean) /
      te_target_sds$targets_sd,
    receptions = (receptions - te_target_means$receptions_mean) /
      te_target_sds$receptions_sd,
    receiving_yards = (receiving_yards - te_target_means$receiving_yards_mean) /
      te_target_sds$receiving_yards_sd
  )

te_2026_targets_scaled <- te_2026_targets %>%
  mutate(
    targets = (targets - te_target_means$targets_mean) /
      te_target_sds$targets_sd,
    receptions = (receptions - te_target_means$receptions_mean) /
      te_target_sds$receptions_sd,
    receiving_yards = (receiving_yards - te_target_means$receiving_yards_mean) /
      te_target_sds$receiving_yards_sd
  )

summary(te_train_targets_scaled)

summary(te_validation_targets_scaled)

summary(te_2026_targets_scaled)

# TE input tensors
te_x_train <- torch_tensor(
  as.matrix(te_train_processed),
  dtype = torch_float()
)

te_x_validation <- torch_tensor(
  as.matrix(te_validation_processed),
  dtype = torch_float()
)

te_x_2026 <- torch_tensor(
  as.matrix(te_2026_processed),
  dtype = torch_float()
)

# TE target tensors: all four targets together
te_y_train <- torch_tensor(
  as.matrix(te_train_targets_scaled),
  dtype = torch_float()
)

te_y_validation <- torch_tensor(
  as.matrix(te_validation_targets_scaled),
  dtype = torch_float()
)

te_y_2026 <- torch_tensor(
  as.matrix(te_2026_targets_scaled),
  dtype = torch_float()
)

te_x_train$shape
te_x_validation$shape
te_x_2026$shape

te_y_train$shape
te_y_validation$shape
te_y_2026$shape

te_model <- nn_module(
  "TEModel",

  initialize = function() {
    self$fc1 <- nn_linear(59, 128)
    self$dropout <- nn_dropout(p = 0.5)

    self$targets_output <- nn_linear(128, 1)
    self$receptions_output <- nn_linear(128, 1)
    self$yards_output <- nn_linear(128, 1)
    self$td_output <- nn_linear(128, 1)
  },

  forward = function(x) {
    x <- self$fc1(x)
    x <- nnf_relu(x)
    x <- self$dropout(x)

    targets <- self$targets_output(x)
    receptions <- self$receptions_output(x)
    yards <- self$yards_output(x)
    td <- self$td_output(x)

    list(
      targets = targets,
      receptions = receptions,
      yards = yards,
      td = td
    )
  }
)

te_net <- te_model()

print(te_net)

te_test_output <- te_net(te_x_train)

te_test_output$targets$shape
te_test_output$receptions$shape
te_test_output$yards$shape
te_test_output$td$shape

te_mse_loss <- nn_mse_loss()

te_td_loss <- nn_bce_with_logits_loss()

te_optimizer <- optim_adam(
  te_net$parameters,
  lr = 0.001
)

te_combined_loss <- function(predictions, targets) {

  targets_loss <- te_mse_loss(
    predictions$targets,
    targets$narrow(2, 1, 1)
  )

  receptions_loss <- te_mse_loss(
    predictions$receptions,
    targets$narrow(2, 2, 1)
  )

  yards_loss <- te_mse_loss(
    predictions$yards,
    targets$narrow(2, 3, 1)
  )

  td_loss <- te_td_loss(
    predictions$td,
    targets$narrow(2, 4, 1)
  )

  regression_loss <- (
    targets_loss +
    receptions_loss +
    yards_loss
  ) / 3

  total_loss <- regression_loss + td_loss

  list(
    total = total_loss,
    targets = targets_loss,
    receptions = receptions_loss,
    yards = yards_loss,
    td = td_loss
  )
}

te_test_losses <- te_combined_loss(
  te_test_output,
  te_y_train
)

te_test_losses$total$item()
te_test_losses$targets$item()
te_test_losses$receptions$item()
te_test_losses$yards$item()
te_test_losses$td$item()

te_epochs <- 30

te_training_losses <- numeric(te_epochs)

for (epoch in 1:te_epochs) {

  te_net$train()

  te_predictions <- te_net(te_x_train)

  te_losses <- te_combined_loss(
    te_predictions,
    te_y_train
  )

  te_optimizer$zero_grad()

  te_losses$total$backward()

  te_optimizer$step()

  te_training_losses[epoch] <- te_losses$total$item()

  cat(
    "Epoch:", epoch,
    "| Loss:", round(te_training_losses[epoch], 4),
    "\n"
  )
}

te_net$eval()

with_no_grad({

  te_validation_predictions <- te_net(te_x_validation)

  te_validation_losses <- te_combined_loss(
    te_validation_predictions,
    te_y_validation
  )

})

te_validation_losses$total$item()
te_validation_losses$targets$item()
te_validation_losses$receptions$item()
te_validation_losses$yards$item()
te_validation_losses$td$item()

te_target_means
te_target_sds

te_validation_targets_pred <- as_array(
  te_validation_predictions$targets
) * te_target_sds$targets_sd +
  te_target_means$targets_mean

te_validation_receptions_pred <- as_array(
  te_validation_predictions$receptions
) * te_target_sds$receptions_sd +
  te_target_means$receptions_mean

te_validation_yards_pred <- as_array(
  te_validation_predictions$yards
) * te_target_sds$receiving_yards_sd +
  te_target_means$receiving_yards_mean

te_validation_td_prob <- as_array(
  torch_sigmoid(te_validation_predictions$td)
)

head(te_validation_targets_pred)
head(te_validation_receptions_pred)
head(te_validation_yards_pred)
head(te_validation_td_prob)

te_validation_mae <- tibble(
  Model = c("Targets", "Receptions", "Receiving Yards"),

  MAE = c(
    mean(abs(te_validation_data$targets -
      as.numeric(te_validation_targets_pred))),

    mean(abs(te_validation_data$receptions -
      as.numeric(te_validation_receptions_pred))),

    mean(abs(te_validation_data$receiving_yards -
      as.numeric(te_validation_yards_pred)))
  )
)

te_validation_mae

te_validation_td_actual <- te_validation_data$any_receiving_td

te_validation_td_pred <- ifelse(
  te_validation_td_prob >= 0.5,
  1,
  0
)

te_td_confusion <- table(
  Actual = te_validation_td_actual,
  Predicted = te_validation_td_pred
)

te_td_confusion

te_td_accuracy <- mean(
  te_validation_td_actual == te_validation_td_pred
)

te_td_precision <- 0

te_td_recall <- 0

te_td_f1 <- 0

te_td_metrics <- tibble(
  Metric = c("Accuracy", "Precision", "Recall", "F1 Score"),
  Value = c(
    te_td_accuracy,
    te_td_precision,
    te_td_recall,
    te_td_f1
  )
)

te_td_metrics

te_td_brier <- mean(
  (te_validation_td_prob -
    te_validation_td_actual)^2
)

te_td_auc_data <- tibble(
  actual = factor(
    te_validation_td_actual,
    levels = c(0, 1)
  ),
  td_probability = as.numeric(te_validation_td_prob)
)

te_td_auc <- yardstick::roc_auc(
  te_td_auc_data,
  truth = actual,
  td_probability,
  event_level = "second"
)

te_td_brier
te_td_auc

te_td_prevalence <- mean(te_validation_td_actual)

te_td_baseline_brier <- mean(
  (te_td_prevalence - te_validation_td_actual)^2
)

te_td_prevalence
te_td_baseline_brier

te_net$eval()

with_no_grad({

  te_2026_predictions <- te_net(te_x_2026)

})

te_2026_targets_pred <- as_array(
  te_2026_predictions$targets
) * te_target_sds$targets_sd +
  te_target_means$targets_mean

te_2026_receptions_pred <- as_array(
  te_2026_predictions$receptions
) * te_target_sds$receptions_sd +
  te_target_means$receptions_mean

te_2026_yards_pred <- as_array(
  te_2026_predictions$yards
) * te_target_sds$receiving_yards_sd +
  te_target_means$receiving_yards_mean

te_2026_td_prob <- as_array(
  torch_sigmoid(te_2026_predictions$td)
)

head(te_2026_targets_pred)
head(te_2026_receptions_pred)
head(te_2026_yards_pred)
head(te_2026_td_prob)

names(te_2026_data)

te_2026_comparison <- tibble(
  Model = c(
    "Neural Network",
    "3-Game Average Baseline"
  ),

  MAE = c(
    mean(abs(
      te_2026_data$receiving_yards -
        as.numeric(te_2026_yards_pred)
    ), na.rm = TRUE),

    mean(abs(
      te_2026_data$receiving_yards -
        te_2026_data$avg_receiving_yards_3
    ), na.rm = TRUE)
  )
)

te_2026_comparison

torch_save(te_net, "te_model.pt")

saveRDS(te_prep, "te_recipe.rds")

te_2026_predictions_table <- te_2026_data %>%
  select(
    player_display_name,
    team,
    week
  ) %>%
  mutate(
    predicted_targets = as.numeric(te_2026_targets_pred),
    predicted_receptions = as.numeric(te_2026_receptions_pred),
    predicted_receiving_yards = as.numeric(te_2026_yards_pred),
    td_probability = as.numeric(te_2026_td_prob)
  ) %>%
  arrange(week, desc(predicted_receiving_yards))

te_2026_predictions_table
View(te_2026_predictions_table)

te_2026_predictions_table <- te_2026_data %>%
  select(
    player_display_name,
    team,
    opponent_team,
    week,
    targets,
    receptions,
    receiving_yards,
    receiving_tds
  ) %>%
  mutate(
    predicted_targets = round(as.numeric(te_2026_targets_pred), 1),
    predicted_receptions = round(as.numeric(te_2026_receptions_pred), 1),
    predicted_receiving_yards = round(as.numeric(te_2026_yards_pred), 1),
    predicted_td_probability = round(as.numeric(te_2026_td_prob), 3)
  ) %>%
  arrange(week, desc(predicted_receiving_yards))

View(te_2026_predictions_table)

# Function to calculate MAE and RMSE
calculate_metrics <- function(actual, predicted) {

  valid_rows <- complete.cases(actual, predicted)

  actual <- actual[valid_rows]
  predicted <- predicted[valid_rows]

  mae <- mean(abs(actual - predicted))
  rmse <- sqrt(mean((actual - predicted)^2))

  c(MAE = mae, RMSE = rmse)
}

# Calculate metrics for each prediction
te_2026_metrics <- tibble(
  Model = c(
    "Targets",
    "Receptions",
    "Receiving Yards (Neural Network)",
    "Receiving Yards (3-Game Baseline)"
  ),

  MAE = c(
    calculate_metrics(
      te_2026_data$targets,
      as.numeric(te_2026_targets_pred)
    )["MAE"],

    calculate_metrics(
      te_2026_data$receptions,
      as.numeric(te_2026_receptions_pred)
    )["MAE"],

    calculate_metrics(
      te_2026_data$receiving_yards,
      as.numeric(te_2026_yards_pred)
    )["MAE"],

    calculate_metrics(
      te_2026_data$receiving_yards,
      te_2026_data$avg_receiving_yards_3
    )["MAE"]
  ),

  RMSE = c(
    calculate_metrics(
      te_2026_data$targets,
      as.numeric(te_2026_targets_pred)
    )["RMSE"],

    calculate_metrics(
      te_2026_data$receptions,
      as.numeric(te_2026_receptions_pred)
    )["RMSE"],

    calculate_metrics(
      te_2026_data$receiving_yards,
      as.numeric(te_2026_yards_pred)
    )["RMSE"],

    calculate_metrics(
      te_2026_data$receiving_yards,
      te_2026_data$avg_receiving_yards_3
    )["RMSE"]
  )
)

te_2026_metrics

# Export 2026 TE predictions to CSV
write.csv(
  te_2026_predictions_table,
  "TE_2026_Neural_Network_Predictions.csv",
  row.names = FALSE
)
