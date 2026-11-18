# ==========================================
# QB PREDICTIONS MODEL
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

qb_data <- load_player_stats(
  seasons = 2020:2026,
  summary_level = "week"
) %>%
  filter(position == "QB")

dim(qb_data)

qb_data %>%
  count(season, season_type)

head(qb_data)

qb_data <- qb_data %>%
  filter(season_type == "REG")

dim(qb_data)

qb_data %>%
  count(season)

summary(qb_data$passing_yards)
summary(qb_data$passing_tds)
summary(qb_data$attempts)
summary(qb_data$completions)

qb_history <- qb_data %>%
  arrange(player_id, season, week) %>%
  group_by(player_id) %>%
  mutate(
    # Previous game statistics
    prev_passing_yards = lag(passing_yards),
    prev_passing_tds = lag(passing_tds),
    prev_attempts = lag(attempts),
    prev_completions = lag(completions),
    prev_interceptions = lag(passing_interceptions),
    prev_sacks = lag(sacks_suffered),
    prev_rushing_yards = lag(rushing_yards),
    prev_carries = lag(carries),
    prev_rushing_tds = lag(rushing_tds),

    # Rolling 3-game averages
    avg_passing_yards_3 = slider::slide_dbl(
      lag(passing_yards), mean, .before = 2, .complete = FALSE
    ),
    avg_passing_tds_3 = slider::slide_dbl(
      lag(passing_tds), mean, .before = 2, .complete = FALSE
    ),
    avg_attempts_3 = slider::slide_dbl(
      lag(attempts), mean, .before = 2, .complete = FALSE
    ),
    avg_completions_3 = slider::slide_dbl(
      lag(completions), mean, .before = 2, .complete = FALSE
    ),

    # Rolling 5-game averages
    avg_passing_yards_5 = slider::slide_dbl(
      lag(passing_yards), mean, .before = 4, .complete = FALSE
    ),
    avg_passing_tds_5 = slider::slide_dbl(
      lag(passing_tds), mean, .before = 4, .complete = FALSE
    ),
    avg_attempts_5 = slider::slide_dbl(
      lag(attempts), mean, .before = 4, .complete = FALSE
    ),
    avg_completions_5 = slider::slide_dbl(
      lag(completions), mean, .before = 4, .complete = FALSE
    ),

    # Previous touchdown indicator
    prev_td_indicator = as.integer(prev_passing_tds > 0),

    # Rolling touchdown rates
    td_rate_last_3 = slider::slide_dbl(
      lag(passing_tds > 0), mean, .before = 2, .complete = FALSE
    ),
    td_rate_last_5 = slider::slide_dbl(
      lag(passing_tds > 0), mean, .before = 4, .complete = FALSE
    )
  ) %>%
  ungroup()

qb_history %>%
  select(
    player_display_name,
    season,
    week,
    passing_yards,
    prev_passing_yards,
    avg_passing_yards_3,
    avg_passing_yards_5,
    td_rate_last_3,
    td_rate_last_5
  ) %>%
  filter(player_display_name == "Josh Allen") %>%
  head(10)

qb_history <- qb_history %>%
  group_by(player_id) %>%
  arrange(season, week, .by_group = TRUE) %>%
  mutate(
    avg_rushing_yards_3 = slider::slide_dbl(
      lag(rushing_yards), mean, .before = 2, .complete = FALSE
    ),
    avg_rushing_yards_5 = slider::slide_dbl(
      lag(rushing_yards), mean, .before = 4, .complete = FALSE
    ),
    avg_carries_3 = slider::slide_dbl(
      lag(carries), mean, .before = 2, .complete = FALSE
    ),
    avg_carries_5 = slider::slide_dbl(
      lag(carries), mean, .before = 4, .complete = FALSE
    )
  ) %>%
  ungroup()

qb_history %>%
  select(
    player_display_name,
    season,
    week,
    rushing_yards,
    prev_rushing_yards,
    avg_rushing_yards_3,
    avg_rushing_yards_5,
    avg_carries_3,
    avg_carries_5
  ) %>%
  filter(player_display_name == "Josh Allen") %>%
  head(10)

qb_team_stats <- load_team_stats(
  seasons = 2020:2026,
  summary_level = "week"
) %>%
  filter(season_type == "REG")

dim(qb_team_stats)

names(qb_team_stats)

qb_defense_history <- qb_team_stats %>%
  arrange(team, season, week) %>%
  group_by(team) %>%
  mutate(
    # Previous game defensive statistics
    prev_pass_yards_allowed = lag(passing_yards),
    prev_pass_tds_allowed = lag(passing_tds),
    prev_pass_attempts_allowed = lag(attempts),
    prev_pass_completions_allowed = lag(completions),
    prev_rush_yards_allowed = lag(rushing_yards),
    prev_rush_attempts_allowed = lag(carries),
    prev_def_sacks = lag(def_sacks),
    prev_def_interceptions = lag(def_interceptions),

    # Rolling 3-game averages
    avg_pass_yards_allowed_3 = slider::slide_dbl(
      lag(passing_yards), mean, .before = 2, .complete = FALSE
    ),
    avg_pass_tds_allowed_3 = slider::slide_dbl(
      lag(passing_tds), mean, .before = 2, .complete = FALSE
    ),
    avg_rush_yards_allowed_3 = slider::slide_dbl(
      lag(rushing_yards), mean, .before = 2, .complete = FALSE
    ),
    avg_def_sacks_3 = slider::slide_dbl(
      lag(def_sacks), mean, .before = 2, .complete = FALSE
    ),
    avg_def_interceptions_3 = slider::slide_dbl(
      lag(def_interceptions), mean, .before = 2, .complete = FALSE
    ),

    # Rolling 5-game averages
    avg_pass_yards_allowed_5 = slider::slide_dbl(
      lag(passing_yards), mean, .before = 4, .complete = FALSE
    ),
    avg_pass_tds_allowed_5 = slider::slide_dbl(
      lag(passing_tds), mean, .before = 4, .complete = FALSE
    ),
    avg_rush_yards_allowed_5 = slider::slide_dbl(
      lag(rushing_yards), mean, .before = 4, .complete = FALSE
    ),
    avg_def_sacks_5 = slider::slide_dbl(
      lag(def_sacks), mean, .before = 4, .complete = FALSE
    ),
    avg_def_interceptions_5 = slider::slide_dbl(
      lag(def_interceptions), mean, .before = 4, .complete = FALSE
    )
  ) %>%
  ungroup()

qb_defense_history %>%
  select(
    team, season, week,
    def_sacks, prev_def_sacks, avg_def_sacks_3, avg_def_sacks_5,
    def_interceptions, prev_def_interceptions,
    avg_def_interceptions_3, avg_def_interceptions_5
  ) %>%
  filter(team == "LA") %>%
  head(10)

qb_model_data <- qb_history %>%
  left_join(
    qb_defense_history %>%
      select(
        game_id,
        team,
        prev_pass_yards_allowed,
        prev_pass_tds_allowed,
        prev_pass_attempts_allowed,
        prev_pass_completions_allowed,
        prev_rush_yards_allowed,
        prev_rush_attempts_allowed,
        prev_def_sacks,
        prev_def_interceptions,
        avg_pass_yards_allowed_3,
        avg_pass_tds_allowed_3,
        avg_rush_yards_allowed_3,
        avg_def_sacks_3,
        avg_def_interceptions_3,
        avg_pass_yards_allowed_5,
        avg_pass_tds_allowed_5,
        avg_rush_yards_allowed_5,
        avg_def_sacks_5,
        avg_def_interceptions_5
      ),
    by = c("game_id", "opponent_team" = "team")
  )

qb_model_data %>%
  select(
    player_display_name,
    season,
    week,
    opponent_team,
    avg_pass_yards_allowed_3,
    avg_rush_yards_allowed_3,
    avg_def_sacks_3,
    avg_def_interceptions_3
  ) %>%
  filter(player_display_name == "Matthew Stafford") %>%
  head(10)

sum(is.na(qb_model_data$avg_pass_yards_allowed_3))

qb_model_data %>%
  filter(is.na(avg_pass_yards_allowed_3)) %>%
  count(season, week)

qb_model_data %>%
  summarise(
    missing_pass = sum(is.na(avg_pass_yards_allowed_3)),
    missing_rush = sum(is.na(avg_rush_yards_allowed_3)),
    missing_sacks = sum(is.na(avg_def_sacks_3)),
    missing_interceptions = sum(is.na(avg_def_interceptions_3))
  )

qb_model_data %>%
  summarise(
    missing_attempts = sum(is.na(attempts)),
    missing_completions = sum(is.na(completions)),
    missing_passing_yards = sum(is.na(passing_yards)),
    missing_passing_tds = sum(is.na(passing_tds)),
    missing_rushing_yards = sum(is.na(rushing_yards))
  )

dim(qb_model_data)

qb_train <- qb_model_data %>%
  filter(season >= 2020, season <= 2024)

qb_validation <- qb_model_data %>%
  filter(season == 2025)

qb_test <- qb_model_data %>%
  filter(season == 2026)

dim(qb_train)
dim(qb_validation)
dim(qb_test)

qb_model_data %>%
  group_by(season) %>%
  summarise(records = n())

qb_predictors <- c(
  # Previous game statistics
  "prev_passing_yards",
  "prev_passing_tds",
  "prev_attempts",
  "prev_completions",
  "prev_interceptions",
  "prev_sacks",
  "prev_rushing_yards",
  "prev_carries",
  "prev_rushing_tds",

  # Passing rolling averages
  "avg_passing_yards_3",
  "avg_passing_yards_5",
  "avg_passing_tds_3",
  "avg_passing_tds_5",
  "avg_attempts_3",
  "avg_attempts_5",
  "avg_completions_3",
  "avg_completions_5",

  # Rushing rolling averages
  "avg_rushing_yards_3",
  "avg_rushing_yards_5",
  "avg_carries_3",
  "avg_carries_5",

  # Touchdown history
  "prev_td_indicator",
  "td_rate_last_3",
  "td_rate_last_5",

  # Opponent defensive passing and rushing averages
  "prev_pass_yards_allowed",
  "prev_pass_tds_allowed",
  "prev_pass_attempts_allowed",
  "prev_pass_completions_allowed",
  "prev_rush_yards_allowed",
  "prev_rush_attempts_allowed",
  "avg_pass_yards_allowed_3",
  "avg_pass_tds_allowed_3",
  "avg_rush_yards_allowed_3",
  "avg_pass_yards_allowed_5",
  "avg_pass_tds_allowed_5",
  "avg_rush_yards_allowed_5",

  # Opponent defensive sacks and interceptions
  "prev_def_sacks",
  "prev_def_interceptions",
  "avg_def_sacks_3",
  "avg_def_interceptions_3",
  "avg_def_sacks_5",
  "avg_def_interceptions_5"
)

qb_predictors %in% names(qb_model_data)

length(qb_predictors)

qb_recipe <- recipe(
  ~ .,
  data = qb_train %>%
    select(all_of(qb_predictors))
) %>%
  step_indicate_na(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

qb_recipe_prep <- prep(
  qb_recipe,
  training = qb_train %>%
    select(all_of(qb_predictors))
)

qb_train_processed <- bake(
  qb_recipe_prep,
  new_data = qb_train %>%
    select(all_of(qb_predictors))
)

dim(qb_train_processed)

qb_validation_processed <- bake(
  qb_recipe_prep,
  new_data = qb_validation %>%
    select(all_of(qb_predictors))
)

qb_test_processed <- bake(
  qb_recipe_prep,
  new_data = qb_test %>%
    select(all_of(qb_predictors))
)

dim(qb_train_processed)
dim(qb_validation_processed)
dim(qb_test_processed)

sum(is.na(qb_train_processed))
sum(is.na(qb_validation_processed))
sum(is.na(qb_test_processed))

qb_targets <- c(
  "attempts",
  "completions",
  "passing_yards",
  "passing_tds",
  "rushing_yards"
)

qb_target_means <- sapply(
  qb_train %>% select(all_of(qb_targets)),
  mean
)

qb_target_sds <- sapply(
  qb_train %>% select(all_of(qb_targets)),
  sd
)

qb_train_targets <- scale(
  qb_train %>% select(all_of(qb_targets)),
  center = qb_target_means,
  scale = qb_target_sds
)

qb_validation_targets <- scale(
  qb_validation %>% select(all_of(qb_targets)),
  center = qb_target_means,
  scale = qb_target_sds
)

qb_test_targets <- scale(
  qb_test %>% select(all_of(qb_targets)),
  center = qb_target_means,
  scale = qb_target_sds
)

colMeans(qb_train_targets)
apply(qb_train_targets, 2, sd)

# Training tensors
qb_x_train <- torch_tensor(
  as.matrix(qb_train_processed),
  dtype = torch_float()
)

qb_y_train <- torch_tensor(
  as.matrix(qb_train_targets),
  dtype = torch_float()
)

# Validation tensors
qb_x_validation <- torch_tensor(
  as.matrix(qb_validation_processed),
  dtype = torch_float()
)

qb_y_validation <- torch_tensor(
  as.matrix(qb_validation_targets),
  dtype = torch_float()
)

# Testing tensors
qb_x_test <- torch_tensor(
  as.matrix(qb_test_processed),
  dtype = torch_float()
)

qb_y_test <- torch_tensor(
  as.matrix(qb_test_targets),
  dtype = torch_float()
)

qb_x_train$size()
qb_y_train$size()

qb_x_validation$size()
qb_y_validation$size()

qb_x_test$size()
qb_y_test$size()

qb_model <- nn_module(
  initialize = function(n_features) {

    self$fc1 <- nn_linear(n_features, 128)

    self$dropout <- nn_dropout(p = 0.5)

    # Five output heads
    self$attempts_head <- nn_linear(128, 1)
    self$completions_head <- nn_linear(128, 1)
    self$passing_yards_head <- nn_linear(128, 1)
    self$passing_tds_head <- nn_linear(128, 1)
    self$rushing_yards_head <- nn_linear(128, 1)
  },

  forward = function(x) {

    x <- self$fc1(x)
    x <- nnf_relu(x)
    x <- self$dropout(x)

    attempts <- self$attempts_head(x)
    completions <- self$completions_head(x)
    passing_yards <- self$passing_yards_head(x)
    passing_tds <- self$passing_tds_head(x)
    rushing_yards <- self$rushing_yards_head(x)

    list(
      attempts,
      completions,
      passing_yards,
      passing_tds,
      rushing_yards
    )
  }
)

qb_net <- qb_model(n_features = 84)

qb_net

qb_loss_fn <- function(predictions, targets) {

  attempts_loss <- nnf_mse_loss(
    predictions[[1]], targets[, 1, drop = FALSE]
  )

  completions_loss <- nnf_mse_loss(
    predictions[[2]], targets[, 2, drop = FALSE]
  )

  passing_yards_loss <- nnf_mse_loss(
    predictions[[3]], targets[, 3, drop = FALSE]
  )

  passing_tds_loss <- nnf_mse_loss(
    predictions[[4]], targets[, 4, drop = FALSE]
  )

  rushing_yards_loss <- nnf_mse_loss(
    predictions[[5]], targets[, 5, drop = FALSE]
  )

  total_loss <- (
    attempts_loss +
    completions_loss +
    passing_yards_loss +
    passing_tds_loss +
    rushing_yards_loss
  ) / 5

  return(total_loss)
}

qb_net$train()

qb_test_output <- qb_net(qb_x_train)

qb_initial_loss <- qb_loss_fn(
  qb_test_output,
  qb_y_train
)

qb_initial_loss$item()

qb_optimizer <- optim_adam(
  params = qb_net$parameters,
  lr = 0.001
)

qb_train_losses <- c()

for (epoch in 1:30) {

  qb_net$train()

  qb_optimizer$zero_grad()

  predictions <- qb_net(qb_x_train)

  loss <- qb_loss_fn(
    predictions,
    qb_y_train
  )

  loss$backward()

  qb_optimizer$step()

  qb_train_losses <- c(
    qb_train_losses,
    loss$item()
  )

  cat(
    sprintf(
      "Epoch %d: Training Loss = %.4f\n",
      epoch,
      loss$item()
    )
  )
}

qb_net$eval()

qb_validation_predictions <- with_no_grad({
  qb_net(qb_x_validation)
})

qb_validation_pred_matrix <- torch_cat(
  qb_validation_predictions,
  dim = 2
)

qb_validation_pred_scaled <- as_array(
  qb_validation_pred_matrix
)

qb_validation_pred <- sweep(
  qb_validation_pred_scaled,
  2,
  qb_target_sds,
  "*"
)

qb_validation_pred <- sweep(
  qb_validation_pred,
  2,
  qb_target_means,
  "+"
)

colnames(qb_validation_pred) <- qb_targets

head(qb_validation_pred)

qb_validation_actual <- as.matrix(
  qb_validation %>%
    select(all_of(qb_targets))
)

qb_validation_mae <- sapply(seq_along(qb_targets), function(i) {
  mean(abs(
    qb_validation_actual[, i] - qb_validation_pred[, i]
  ))
})

qb_validation_rmse <- sapply(seq_along(qb_targets), function(i) {
  sqrt(mean(
    (qb_validation_actual[, i] - qb_validation_pred[, i])^2
  ))
})

qb_validation_metrics <- data.frame(
  target = qb_targets,
  MAE = qb_validation_mae,
  RMSE = qb_validation_rmse
)

qb_validation_metrics

qb_validation_baseline <- data.frame(
  target = qb_targets,
  MAE = c(
    mean(abs(qb_validation$attempts - qb_validation$avg_attempts_3), na.rm = TRUE),
    mean(abs(qb_validation$completions - qb_validation$avg_completions_3), na.rm = TRUE),
    mean(abs(qb_validation$passing_yards - qb_validation$avg_passing_yards_3), na.rm = TRUE),
    mean(abs(qb_validation$passing_tds - qb_validation$avg_passing_tds_3), na.rm = TRUE),
    mean(abs(qb_validation$rushing_yards - qb_validation$avg_rushing_yards_3), na.rm = TRUE)
  )
)

qb_validation_comparison <- data.frame(
  target = qb_targets,
  Neural_Network_MAE = qb_validation_mae,
  Baseline_MAE = qb_validation_baseline$MAE
)

qb_validation_comparison

qb_net$eval()

qb_test_predictions <- with_no_grad({
  qb_net(qb_x_test)
})

qb_test_pred_matrix <- torch_cat(
  qb_test_predictions,
  dim = 2
)

qb_test_pred_scaled <- as_array(qb_test_pred_matrix)

qb_test_pred <- sweep(
  qb_test_pred_scaled,
  2,
  qb_target_sds,
  "*"
)

qb_test_pred <- sweep(
  qb_test_pred,
  2,
  qb_target_means,
  "+"
)

colnames(qb_test_pred) <- qb_targets

head(qb_test_pred)

qb_test_actual <- as.matrix(
  qb_test %>%
    select(all_of(qb_targets))
)

qb_test_mae <- sapply(seq_along(qb_targets), function(i) {
  mean(abs(
    qb_test_actual[, i] - qb_test_pred[, i]
  ))
})

qb_test_rmse <- sapply(seq_along(qb_targets), function(i) {
  sqrt(mean(
    (qb_test_actual[, i] - qb_test_pred[, i])^2
  ))
})

qb_test_metrics <- data.frame(
  target = qb_targets,
  MAE = qb_test_mae,
  RMSE = qb_test_rmse
)

qb_test_metrics

qb_test_baseline <- data.frame(
  target = qb_targets,
  MAE = c(
    mean(abs(qb_test$attempts - qb_test$avg_attempts_3), na.rm = TRUE),
    mean(abs(qb_test$completions - qb_test$avg_completions_3), na.rm = TRUE),
    mean(abs(qb_test$passing_yards - qb_test$avg_passing_yards_3), na.rm = TRUE),
    mean(abs(qb_test$passing_tds - qb_test$avg_passing_tds_3), na.rm = TRUE),
    mean(abs(qb_test$rushing_yards - qb_test$avg_rushing_yards_3), na.rm = TRUE)
  )
)

qb_test_comparison <- data.frame(
  target = qb_targets,
  Neural_Network_MAE = qb_test_mae,
  Baseline_MAE = qb_test_baseline$MAE
)

qb_test_comparison

qb_final_results <- data.frame(
  target = qb_targets,
  Validation_MAE = qb_validation_mae,
  Validation_RMSE = qb_validation_rmse,
  Validation_Baseline_MAE = qb_validation_baseline$MAE,
  Test_MAE = qb_test_mae,
  Test_RMSE = qb_test_rmse,
  Test_Baseline_MAE = qb_test_baseline$MAE
)

qb_final_results

names(qb_test)[
  names(qb_test) %in% c(
    "player_display_name",
    "player_name",
    "recent_team",
    "team",
    "opponent_team",
    "week",
    "season"
  )
]

qb_2026_predictions <- qb_test %>%
  select(
    player_display_name,
    season,
    week,
    team,
    opponent_team
  ) %>%
  bind_cols(
    as.data.frame(qb_test_pred)
  ) %>%
  arrange(week, player_display_name)

head(qb_2026_predictions, 20)

qb_2026_comparison <- qb_test %>%
  select(
    player_display_name,
    season,
    week,
    team,
    opponent_team,
    attempts,
    completions,
    passing_yards,
    passing_tds,
    rushing_yards
  ) %>%
  bind_cols(
    as.data.frame(qb_test_pred) %>%
      rename(
        Predicted_Attempts = attempts,
        Predicted_Completions = completions,
        Predicted_Passing_Yards = passing_yards,
        Predicted_Passing_TDs = passing_tds,
        Predicted_Rushing_Yards = rushing_yards
      )
  ) %>%
  rename(
    Actual_Attempts = attempts,
    Actual_Completions = completions,
    Actual_Passing_Yards = passing_yards,
    Actual_Passing_TDs = passing_tds,
    Actual_Rushing_Yards = rushing_yards
  ) %>%
  arrange(week, player_display_name)

View(qb_2026_comparison)
