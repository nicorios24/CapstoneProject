# ==========================================
# RB XGBOOST MODEL
# Predicting NFL Running Back Rushing Yards
# ==========================================

# Load packages

library(tidyverse)
library(nflreadr)
library(xgboost)
library(slider)
library(lubridate)

# Reproducibility

set.seed(123)

# ==========================================
# STEP 2: LOAD NFL PLAYER STATISTICS
# ==========================================

# Download player statistics for 2020–2025

player_stats <- load_player_stats(2020:2026)

# Keep regular-season statistics

player_data <- player_stats |>
  filter(season_type == "REG")

# Check dataset dimensions

dim(player_data)

# Review available columns

names(player_data)

# Check seasons included

player_data |>
  count(season)

# Review running back records

rb_data <- player_data |>
  filter(position == "RB")

dim(rb_data)

head(rb_data)

# ==========================================
# STEP 3: INSPECT RUNNING BACK DATA
# ==========================================

# Display all available columns
names(rb_data)

# Check the number of records per season
rb_data |>
  count(season)

# Preview a few records
rb_data |>
  select(season, week, player_name, opponent_team,
         rushing_yards, carries) |>
  head(10)

# ==========================================
# STEP 4: CREATE PLAYER-HISTORY FEATURES
# ==========================================

rb_history <- rb_data |>
  arrange(season, week) |>
  group_by(player_id) |>
  mutate(
    # Previous game statistics
    prev_rushing_yards = lag(rushing_yards),
    prev_carries = lag(carries),
    prev_rushing_tds = lag(rushing_tds),

    prev_receiving_yards = lag(receiving_yards),
    prev_receptions = lag(receptions),
    prev_targets = lag(targets),

    # Average rushing yards over previous games
    avg_rushing_yards_3 = slider::slide_dbl(
      lag(rushing_yards),
      mean,
      .before = 2,
      .complete = TRUE
    ),

    avg_rushing_yards_5 = slider::slide_dbl(
      lag(rushing_yards),
      mean,
      .before = 4,
      .complete = TRUE
    ),

    # Average carries over previous games
    avg_carries_3 = slider::slide_dbl(
      lag(carries),
      mean,
      .before = 2,
      .complete = TRUE
    ),

    avg_carries_5 = slider::slide_dbl(
      lag(carries),
      mean,
      .before = 4,
      .complete = TRUE
    )
  ) |>
  ungroup()

rb_history |>
  select(
    season, week, player_name,
    rushing_yards, carries,
    prev_rushing_yards, prev_carries,
    avg_rushing_yards_3, avg_rushing_yards_5,
    avg_carries_3, avg_carries_5
  ) |>
  filter(player_name == "D.Henry") |>
  arrange(season, week) |>
  head(10)

# ==========================================
# STEP 5: LOAD WEEKLY TEAM STATISTICS
# ==========================================

team_stats <- load_team_stats(
  seasons = 2020:2026
)

# Inspect dataset dimensions
dim(team_stats)

# Inspect available columns
names(team_stats)

# Preview records
head(team_stats)

# ==========================================
# STEP 6: CREATE OPPONENT-DEFENSE FEATURES
# ==========================================

defense_data <- team_stats |>
  filter(season_type == "REG") |>
  transmute(
    season,
    week,
    defense_team = opponent_team,

    rushing_yards_allowed = rushing_yards,
    rushing_attempts_allowed = carries,
    rushing_tds_allowed = rushing_tds,
    rushing_epa_allowed = rushing_epa
  ) |>
  arrange(season, week) |>
  group_by(defense_team) |>
  mutate(
    # Previous game allowed
    prev_rushing_yards_allowed = lag(rushing_yards_allowed),
    prev_rushing_attempts_allowed = lag(rushing_attempts_allowed),
    prev_rushing_tds_allowed = lag(rushing_tds_allowed),
    prev_rushing_epa_allowed = lag(rushing_epa_allowed),

    # Previous-game yards per carry allowed
    prev_yards_per_carry_allowed = if_else(
      prev_rushing_attempts_allowed > 0,
      prev_rushing_yards_allowed / prev_rushing_attempts_allowed,
      NA_real_
    ),

    # Previous 3-game defensive averages
    avg_rushing_yards_allowed_3 = slider::slide_dbl(
      lag(rushing_yards_allowed),
      mean,
      .before = 2,
      .complete = TRUE
    ),

    avg_rushing_attempts_allowed_3 = slider::slide_dbl(
      lag(rushing_attempts_allowed),
      mean,
      .before = 2,
      .complete = TRUE
    ),

    avg_rushing_tds_allowed_3 = slider::slide_dbl(
      lag(rushing_tds_allowed),
      mean,
      .before = 2,
      .complete = TRUE
    ),

    # Previous 5-game defensive averages
    avg_rushing_yards_allowed_5 = slider::slide_dbl(
      lag(rushing_yards_allowed),
      mean,
      .before = 4,
      .complete = TRUE
    ),

    avg_rushing_attempts_allowed_5 = slider::slide_dbl(
      lag(rushing_attempts_allowed),
      mean,
      .before = 4,
      .complete = TRUE
    ),

    avg_rushing_tds_allowed_5 = slider::slide_dbl(
      lag(rushing_tds_allowed),
      mean,
      .before = 4,
      .complete = TRUE
    )
  ) |>
  ungroup()

defense_data |>
  filter(defense_team == "SF") |>
  select(
    season, week, defense_team,
    rushing_yards_allowed,
    prev_rushing_yards_allowed,
    avg_rushing_yards_allowed_3,
    avg_rushing_yards_allowed_5
  ) |>
  arrange(season, week) |>
  head(10)

# ==========================================
# STEP 7: COMBINE PLAYER AND DEFENSE FEATURES
# ==========================================

rb_model_data <- rb_history |>
  left_join(
    defense_data |>
      select(
        season,
        week,
        defense_team,

        # Previous-game defensive statistics
        starts_with("prev_"),

        # Rolling defensive averages
        starts_with("avg_")
      ),
    by = c(
      "season",
      "week",
      "opponent_team" = "defense_team"
    )
  )

# Check dataset dimensions
dim(rb_model_data)

# Check whether defensive features were matched
rb_model_data |>
  summarise(
    total_rows = n(),
    missing_opponent_average = sum(
      is.na(avg_rushing_yards_allowed_3)
    )
  )

rb_model_data |>
  select(
    season,
    week,
    player_name,
    opponent_team,
    rushing_yards,
    prev_rushing_yards,
    avg_rushing_yards_3,
    avg_rushing_yards_allowed_3,
    avg_rushing_yards_allowed_5,
    prev_yards_per_carry_allowed
  ) |>
  filter(player_name == "D.Henry") |>
  arrange(season, week) |>
  head(10)

# ==========================================
# STEP 8: PLAYER PERFORMANCE VARIABILITY
# ==========================================

rb_history <- rb_history |>
  arrange(season, week) |>
  group_by(player_id) |>
  mutate(
    # Rushing yards variability
    sd_rushing_yards_3 = slider::slide_dbl(
      lag(rushing_yards),
      sd,
      .before = 2,
      .complete = TRUE
    ),

    sd_rushing_yards_5 = slider::slide_dbl(
      lag(rushing_yards),
      sd,
      .before = 4,
      .complete = TRUE
    ),

    # Carries variability
    sd_carries_3 = slider::slide_dbl(
      lag(carries),
      sd,
      .before = 2,
      .complete = TRUE
    ),

    sd_carries_5 = slider::slide_dbl(
      lag(carries),
      sd,
      .before = 4,
      .complete = TRUE
    )
  ) |>
  ungroup()

rb_model_data <- rb_history |>
  left_join(
    defense_data |>
      select(
        season,
        week,
        defense_team,
        starts_with("prev_"),
        starts_with("avg_")
      ),
    by = c(
      "season",
      "week",
      "opponent_team" = "defense_team"
    )
  )

rb_model_data |>
  select(
    season,
    week,
    player_name,
    rushing_yards,
    avg_rushing_yards_3,
    sd_rushing_yards_3,
    avg_carries_3,
    sd_carries_3
  ) |>
  filter(player_name == "D.Henry") |>
  arrange(season, week) |>
  head(10)

# ==========================================
# STEP 9: LOAD NFL SCHEDULE INFORMATION
# ==========================================

schedules <- load_schedules(2020:2026)

# Inspect dimensions
dim(schedules)

# Inspect available columns
names(schedules)

# Preview schedule records
head(schedules)

# ==========================================
# STEP 10: CREATE GAME-CONTEXT FEATURES
# ==========================================

schedule_context <- schedules |>
  filter(game_type == "REG") |>
  select(
    game_id,
    home_team,
    away_team,
    home_rest,
    away_rest,
    roof,
    surface,
    temp,
    wind
  ) |>
  pivot_longer(
    cols = c(home_team, away_team),
    names_to = "team_side",
    values_to = "team"
  ) |>
  mutate(
    is_home = if_else(team_side == "home_team", 1L, 0L),
    rest_days = if_else(
      team_side == "home_team",
      home_rest,
      away_rest
    )
  ) |>
  select(
    game_id,
    team,
    is_home,
    rest_days,
    roof,
    surface,
    temp,
    wind
  )

dim(schedule_context)

head(schedule_context)

schedule_context |>
  count(is_home)

schedule_context |>
  summarise(
    missing_rest = sum(is.na(rest_days)),
    missing_roof = sum(is.na(roof)),
    missing_surface = sum(is.na(surface)),
    missing_temp = sum(is.na(temp)),
    missing_wind = sum(is.na(wind))
  )

# ==========================================
# STEP 11: JOIN GAME-CONTEXT FEATURES
# ==========================================

rb_model_data <- rb_model_data |>
  left_join(
    schedule_context,
    by = c("game_id", "team")
  )

dim(rb_model_data)

rb_model_data |>
  summarise(
    total_rows = n(),
    missing_home_status = sum(is.na(is_home)),
    missing_rest_days = sum(is.na(rest_days)),
    missing_roof = sum(is.na(roof)),
    missing_surface = sum(is.na(surface)),
    missing_temp = sum(is.na(temp)),
    missing_wind = sum(is.na(wind))
  )

rb_model_data |>
  select(
    player_name,
    season,
    week,
    team,
    opponent_team,
    is_home,
    rest_days,
    roof,
    surface,
    temp,
    wind
  ) |>
  head(10)

# ==========================================
# STEP 12: ADD RECEIVING AND TD HISTORY
# ==========================================

rb_history <- rb_history |>
  arrange(season, week) |>
  group_by(player_id) |>
  mutate(
    avg_receiving_yards_3 = slider::slide_dbl(
      lag(receiving_yards), mean, .before = 2, .complete = TRUE
    ),
    avg_receiving_yards_5 = slider::slide_dbl(
      lag(receiving_yards), mean, .before = 4, .complete = TRUE
    ),
    avg_receptions_3 = slider::slide_dbl(
      lag(receptions), mean, .before = 2, .complete = TRUE
    ),
    avg_targets_3 = slider::slide_dbl(
      lag(targets), mean, .before = 2, .complete = TRUE
    ),
    avg_rushing_tds_3 = slider::slide_dbl(
      lag(rushing_tds), mean, .before = 2, .complete = TRUE
    )
  ) |>
  ungroup()

# Rebuild player and defense data
rb_model_data <- rb_history |>
  left_join(
    defense_data |>
      select(
        season,
        week,
        defense_team,
        starts_with("prev_"),
        starts_with("avg_")
      ),
    by = c(
      "season",
      "week",
      "opponent_team" = "defense_team"
    )
  )

# Reattach game-context information
rb_model_data <- rb_model_data |>
  left_join(
    schedule_context,
    by = c("game_id", "team")
  )

rb_model_data |>
  select(
    player_name,
    season,
    week,
    avg_receiving_yards_3,
    avg_receiving_yards_5,
    avg_receptions_3,
    avg_targets_3,
    avg_rushing_tds_3
  ) |>
  head(10)

dim(rb_model_data)

rb_model_data |>
  filter(player_name == "D.Henry", season == 2020) |>
  select(
    player_name,
    week,
    avg_receiving_yards_3,
    avg_receptions_3,
    avg_targets_3,
    avg_rushing_tds_3
  ) |>
  head(10)

rb_model_data |>
  summarise(
    missing_receiving_3 = sum(is.na(avg_receiving_yards_3)),
    missing_receiving_5 = sum(is.na(avg_receiving_yards_5)),
    missing_receptions = sum(is.na(avg_receptions_3)),
    missing_targets = sum(is.na(avg_targets_3)),
    missing_rushing_tds = sum(is.na(avg_rushing_tds_3))
  )

# ==========================================
# STEP 13: BUILD THE MODELING DATASET
# ==========================================

rb_modeling_data <- rb_model_data |>
  select(
    # Target
    rushing_yards,

    # Game information
    season,
    week,
    is_home,
    rest_days,
    roof,
    surface,
    temp,
    wind,

    # Previous player performance
    prev_rushing_yards,
    prev_carries,
    prev_rushing_tds,
    prev_receiving_yards,
    prev_receptions,
    prev_targets,

    # Recent player averages
    avg_rushing_yards_3,
    avg_rushing_yards_5,
    avg_carries_3,
    avg_carries_5,
    avg_receiving_yards_3,
    avg_receiving_yards_5,
    avg_receptions_3,
    avg_targets_3,
    avg_rushing_tds_3,

    # Player consistency
    sd_rushing_yards_3,
    sd_rushing_yards_5,
    sd_carries_3,
    sd_carries_5,

    # Opponent defensive performance
    prev_rushing_yards_allowed,
    prev_rushing_attempts_allowed,
    prev_rushing_tds_allowed,
    prev_rushing_epa_allowed,
    prev_yards_per_carry_allowed,
    avg_rushing_yards_allowed_3,
    avg_rushing_attempts_allowed_3,
    avg_rushing_tds_allowed_3,
    avg_rushing_yards_allowed_5,
    avg_rushing_attempts_allowed_5,
    avg_rushing_tds_allowed_5
  ) |>
  filter(!is.na(rushing_yards))

# Dataset dimensions
dim(rb_modeling_data)

# Check missing target values
sum(is.na(rb_modeling_data$rushing_yards))

# Check target distribution
summary(rb_modeling_data$rushing_yards)

# Check records by season
rb_modeling_data |>
  count(season)

# ==========================================
# STEP 14: SPLIT THE DATA
# ==========================================

# Training data: 2020-2023
train_data <- rb_modeling_data |>
  filter(season >= 2020, season <= 2023)

# Validation data: 2024
validation_data <- rb_modeling_data |>
  filter(season == 2024)

# Final testing data: 2025
test_data <- rb_modeling_data |>
  filter(season == 2025)

# Check dataset sizes
dim(train_data)
dim(validation_data)
dim(test_data)

# Check seasons in each dataset
train_data |> count(season)
validation_data |> count(season)
test_data |> count(season)

# ==========================================
# STEP 15: PREPARE XGBOOST DATA
# ==========================================

# Combine the three datasets
all_data <- bind_rows(
  train_data,
  validation_data,
  test_data
)

# Create predictor data without the target
predictor_data <- all_data |>
  select(-rushing_yards) |>
  mutate(
    roof = as.factor(roof),
    surface = as.factor(surface)
  )

# Preserve rows containing missing values
predictor_formula <- ~ . - 1

predictor_frame <- model.frame(
  predictor_formula,
  data = predictor_data,
  na.action = na.pass
)

# Create numerical predictor matrix
x_all <- model.matrix(
  predictor_formula,
  data = predictor_frame
)

# Separate the predictor matrices
x_train <- x_all[1:nrow(train_data), , drop = FALSE]

x_validation <- x_all[
  (nrow(train_data) + 1):
    (nrow(train_data) + nrow(validation_data)),
  ,
  drop = FALSE
]

x_test <- x_all[
  (nrow(train_data) + nrow(validation_data) + 1):
    nrow(all_data),
  ,
  drop = FALSE
]

# Create target vectors
y_train <- train_data$rushing_yards
y_validation <- validation_data$rushing_yards
y_test <- test_data$rushing_yards

# Check total rows
nrow(x_all)
nrow(all_data)

# Check matrix dimensions
dim(x_train)
dim(x_validation)
dim(x_test)

# Check that all three have identical columns
identical(colnames(x_train), colnames(x_validation))
identical(colnames(x_train), colnames(x_test))

# Check missing values
sum(is.na(x_train))
sum(is.na(x_validation))
sum(is.na(x_test))

# ==========================================
# STEP 16: CREATE XGBOOST DATA OBJECTS
# ==========================================

dtrain <- xgb.DMatrix(
  data = x_train,
  label = y_train,
  missing = NA
)

dvalidation <- xgb.DMatrix(
  data = x_validation,
  label = y_validation,
  missing = NA
)

dtest <- xgb.DMatrix(
  data = x_test,
  label = y_test,
  missing = NA
)

dtrain
dvalidation
dtest

# ==========================================
# STEP 17: TRAIN FIRST XGBOOST MODEL
# ==========================================

# Model parameters
xgb_params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.05,
  max_depth = 4,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 5
)

# Train model
set.seed(123)

rb_xgb_model <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = 1000,
  evals = list(
    train = dtrain,
    validation = dvalidation
  ),
  early_stopping_rounds = 50,
  verbose = 1
)

# ==========================================
# STEP 18: EVALUATE VALIDATION PERFORMANCE
# ==========================================

# Generate predictions
validation_predictions <- predict(
  rb_xgb_model,
  dvalidation
)

# Calculate errors
validation_errors <- y_validation - validation_predictions

# Calculate evaluation metrics
validation_rmse <- sqrt(mean(validation_errors^2))
validation_mae <- mean(abs(validation_errors))

# Display results
validation_results <- tibble(
  RMSE = validation_rmse,
  MAE = validation_mae
)

validation_results

# ==========================================
# STEP 19: COMPARE AGAINST 3-GAME BASELINE
# ==========================================

# Keep validation records with a 3-game average
baseline_rows <- !is.na(validation_data$avg_rushing_yards_3)

# Actual rushing yards
baseline_actual <- validation_data$rushing_yards[baseline_rows]

# Baseline predictions
baseline_predictions <- validation_data$avg_rushing_yards_3[baseline_rows]

# XGBoost predictions on the same records
xgb_predictions_matched <- validation_predictions[baseline_rows]

# Compare performance
baseline_comparison <- tibble(
  Model = c("3-Game Average", "XGBoost"),
  RMSE = c(
    sqrt(mean((baseline_actual - baseline_predictions)^2)),
    sqrt(mean((baseline_actual - xgb_predictions_matched)^2))
  ),
  MAE = c(
    mean(abs(baseline_actual - baseline_predictions)),
    mean(abs(baseline_actual - xgb_predictions_matched))
  )
)

baseline_comparison

# ==========================================
# STEP 20: TUNE XGBOOST PARAMETERS
# ==========================================

tuning_grid <- expand.grid(
  max_depth = c(3, 4, 5),
  eta = c(0.01, 0.03, 0.05)
)

tuning_models <- list()
tuning_results <- tibble()

for (i in seq_len(nrow(tuning_grid))) {

  params <- list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    eta = tuning_grid$eta[i],
    max_depth = tuning_grid$max_depth[i],
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5
  )

  set.seed(123)

  model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 1000,
    evals = list(
      train = dtrain,
      validation = dvalidation
    ),
    early_stopping_rounds = 50,
    verbose = 0
  )

  predictions <- predict(model, dvalidation)

  current_rmse <- sqrt(
    mean((y_validation - predictions)^2)
  )

  tuning_models[[i]] <- model

  tuning_results <- bind_rows(
    tuning_results,
    tibble(
      max_depth = tuning_grid$max_depth[i],
      eta = tuning_grid$eta[i],
      best_iteration = model$best_iteration,
      validation_rmse = current_rmse
    )
  )
}

print(tuning_results, n = Inf)

tuning_results |>
  arrange(validation_rmse)

print(tuning_results, n = Inf, digits = 10)

data.frame(
  max_depth = tuning_results$max_depth,
  eta = tuning_results$eta,
  validation_rmse = sprintf("%.10f", tuning_results$validation_rmse)
)

# ==========================================
# STEP 21: SELECT BEST MODEL
# ==========================================

best_index <- which.min(tuning_results$validation_rmse)

best_rb_xgb_model <- tuning_models[[best_index]]

best_params <- tuning_results[best_index, ]

best_params

# Original model predictions
original_predictions <- predict(
  rb_xgb_model,
  dvalidation
)

# Tuned model predictions
tuned_predictions <- predict(
  best_rb_xgb_model,
  dvalidation
)

# Compare performance
model_comparison <- tibble(
  Model = c("Original XGBoost", "Tuned XGBoost"),
  RMSE = c(
    sqrt(mean((y_validation - original_predictions)^2)),
    sqrt(mean((y_validation - tuned_predictions)^2))
  ),
  MAE = c(
    mean(abs(y_validation - original_predictions)),
    mean(abs(y_validation - tuned_predictions))
  )
)

model_comparison

# ==========================================
# STEP 23: TUNED MODEL VS. BASELINE
# ==========================================

baseline_rows <- !is.na(validation_data$avg_rushing_yards_3)

baseline_actual <- y_validation[baseline_rows]

baseline_predictions <- validation_data$avg_rushing_yards_3[baseline_rows]

tuned_predictions_matched <- tuned_predictions[baseline_rows]

final_validation_comparison <- tibble(
  Model = c("3-Game Average", "Original XGBoost", "Tuned XGBoost"),
  RMSE = c(
    sqrt(mean((baseline_actual - baseline_predictions)^2)),
    sqrt(mean((baseline_actual - original_predictions[baseline_rows])^2)),
    sqrt(mean((baseline_actual - tuned_predictions_matched)^2))
  ),
  MAE = c(
    mean(abs(baseline_actual - baseline_predictions)),
    mean(abs(baseline_actual - original_predictions[baseline_rows])),
    mean(abs(baseline_actual - tuned_predictions_matched))
  )
)

final_validation_comparison

# ==========================================
# STEP 24: SAVE TUNED MODEL
# ==========================================

xgb.save(
  best_rb_xgb_model,
  "best_rb_xgb_model.json"
)

# ==========================================
# STEP 25: FINAL TEST EVALUATION
# ==========================================

test_predictions <- predict(
  best_rb_xgb_model,
  dtest
)

test_rmse <- sqrt(
  mean((y_test - test_predictions)^2)
)

test_mae <- mean(
  abs(y_test - test_predictions)
)

test_results <- tibble(
  Model = "Tuned XGBoost",
  RMSE = test_rmse,
  MAE = test_mae
)

test_results

# ==========================================
# STEP 26: 2025 BASELINE COMPARISON
# ==========================================

test_baseline_rows <- !is.na(test_data$avg_rushing_yards_3)

test_actual <- y_test[test_baseline_rows]

test_baseline_predictions <- 
  test_data$avg_rushing_yards_3[test_baseline_rows]

test_xgb_predictions <- 
  test_predictions[test_baseline_rows]

test_comparison <- tibble(
  Model = c("3-Game Average", "Tuned XGBoost"),
  RMSE = c(
    sqrt(mean((test_actual - test_baseline_predictions)^2)),
    sqrt(mean((test_actual - test_xgb_predictions)^2))
  ),
  MAE = c(
    mean(abs(test_actual - test_baseline_predictions)),
    mean(abs(test_actual - test_xgb_predictions))
  )
)

test_comparison

# ==========================================
# STEP 27: FEATURE IMPORTANCE
# ==========================================

feature_importance <- xgb.importance(
  feature_names = colnames(x_train),
  model = best_rb_xgb_model
)

head(feature_importance, 15)

# ==========================================
# STEP 28: PLOT FEATURE IMPORTANCE
# ==========================================

xgb.plot.importance(
  feature_importance,
  top_n = 15,
  measure = "Gain"
)

# ==========================================
# STEP 29: ANALYZE PREDICTION ERRORS
# ==========================================

test_error_data <- rb_model_data |>
  filter(season == 2025) |>
  select(
    player_name,
    team,
    opponent_team,
    week,
    rushing_yards
  ) |>
  mutate(
    predicted_rushing_yards = test_predictions,
    prediction_error = rushing_yards - predicted_rushing_yards,
    absolute_error = abs(prediction_error)
  )

test_error_data |>
  summarise(
    average_error = mean(prediction_error),
    average_absolute_error = mean(absolute_error),
    largest_error = max(absolute_error),
    overpredictions = sum(prediction_error < 0),
    underpredictions = sum(prediction_error > 0),
    exact_predictions = sum(prediction_error == 0)
  )

# ==========================================
# STEP 31: LARGEST PREDICTION ERRORS
# ==========================================

largest_errors <- test_error_data |>
  arrange(desc(absolute_error)) |>
  select(
    player_name,
    team,
    opponent_team,
    week,
    rushing_yards,
    predicted_rushing_yards,
    prediction_error,
    absolute_error
  ) |>
  slice_head(n = 10)

largest_errors

best_rb_xgb_model$best_iteration

best_rb_xgb_model$niter

# Check installed XGBoost version
packageVersion("xgboost")

# Inspect the model
str(best_rb_xgb_model, max.level = 1)

# Check available model information
names(best_rb_xgb_model)

# Inspect early-stopping information
attr(best_rb_xgb_model, "early_stop")

# Check the evaluation log columns
names(attr(best_rb_xgb_model, "evaluation_log"))

# Display the last 10 training iterations
tail(attr(best_rb_xgb_model, "evaluation_log"), 10)

# ==========================================
# STEP 34: PREPARE FULL TRAINING DATA
# ==========================================

full_training_data <- bind_rows(
  train_data,
  validation_data,
  test_data
)

x_full <- x_all

y_full <- full_training_data$rushing_yards

dtrain_full <- xgb.DMatrix(
  data = x_full,
  label = y_full,
  missing = NA
)

dim(x_full)
length(y_full)

# ==========================================
# STEP 35: TRAIN PRODUCTION MODEL
# ==========================================

production_params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.01,
  max_depth = 5,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 5
)

set.seed(123)

rb_xgb_production <- xgb.train(
  params = production_params,
  data = dtrain_full,
  nrounds = 456,
  verbose = 1
)

xgb.save(
  rb_xgb_production,
  "rb_xgb_production.json"
)

file.exists("rb_xgb_production.json")
loaded_rb_model <- xgb.load("rb_xgb_production.json")

dim(schedules)
names(schedules)
head(schedules)
ls("package:nflreadr", pattern = "roster")

upcoming_schedule <- schedules[
  schedules$season == 2026 &
  schedules$game_type == "REG",
]

dim(upcoming_schedule)
head(upcoming_schedule[, c(
  "week",
  "gameday",
  "away_team",
  "home_team"
)])

schedule_df <- as.data.frame(upcoming_schedule)

schedule_df[
  (nrow(schedule_df) - 19):nrow(schedule_df),
  c("week", "gameday", "away_team", "home_team")
]

rosters_2026 <- load_rosters_weekly(seasons = 2026)
dim(rosters_2026)
names(rosters_2026)

sort(unique(rosters_2026$week))
table(rosters_2026$status, useNA = "ifany")
latest_roster_week <- max(rosters_2026$week, na.rm = TRUE)

rb_roster <- rosters_2026 |>
  dplyr::filter(
    week == latest_roster_week,
    position == "RB"
  ) |>
  dplyr::select(
    full_name,
    team,
    position,
    depth_chart_position,
    status,
    gsis_id
  )

head(rb_roster, 15)

rb_roster_week10 <- rosters_2026 |>
  dplyr::filter(
    week == 9,
    position == "RB",
    depth_chart_position == "RB",
    status == "ACT"
  ) |>
  dplyr::select(
    full_name,
    team,
    depth_chart_position,
    status,
    gsis_id
  ) |>
  dplyr::distinct(gsis_id, .keep_all = TRUE)

nrow(rb_roster_week10)
head(rb_roster_week10, 20)

schedule_df <- as.data.frame(upcoming_schedule)

schedule_df[
  schedule_df$week == 10,
  c("week", "gameday", "away_team", "home_team")
]

# ==========================================
# STEP 43: CREATE WEEK 10 TEAM MATCHUPS
# ==========================================

week10_games <- schedule_df[
  schedule_df$week == 10,
  c("away_team", "home_team")
]

away_matchups <- data.frame(
  team = week10_games$away_team,
  opponent_team = week10_games$home_team,
  is_home = 0
)

home_matchups <- data.frame(
  team = week10_games$home_team,
  opponent_team = week10_games$away_team,
  is_home = 1
)

week10_matchups <- rbind(
  away_matchups,
  home_matchups
)

week10_rb_candidates <- merge(
  as.data.frame(rb_roster_week10),
  week10_matchups,
  by = "team"
)

week10_rb_candidates <- week10_rb_candidates[
  order(week10_rb_candidates$team),
]

nrow(week10_rb_candidates)

head(
  week10_rb_candidates[
    order(week10_rb_candidates$team),
    c("full_name", "team", "opponent_team", "is_home")
  ],
  25
)

table(week10_rb_candidates$team)

# Check the historical player ID column
names(rb_history)

# Inspect a few roster IDs
head(week10_rb_candidates[, c(
  "full_name",
  "gsis_id"
)], 10)

# Inspect a few historical player IDs
rb_history |>
  dplyr::select(
    player_id,
    player_name,
    season,
    week
  ) |>
  head(10)

matched_rb_ids <- week10_rb_candidates |>
  dplyr::inner_join(
    rb_history |>
      dplyr::distinct(player_id),
    by = c("gsis_id" = "player_id")
  )

nrow(matched_rb_ids)

unmatched_rb_ids <- week10_rb_candidates |>
  dplyr::anti_join(
    rb_history |>
      dplyr::distinct(player_id),
    by = c("gsis_id" = "player_id")
  )

unmatched_rb_ids |>
  dplyr::select(full_name, team, gsis_id)

# ==========================================
# STEP 47: GET LATEST PLAYER HISTORY
# ==========================================

latest_rb_history <- rb_history |>
  dplyr::filter(
    season_type == "REG",
    season <= 2026,
    week <= 9
  ) |>
  dplyr::arrange(
    player_id,
    dplyr::desc(season),
    dplyr::desc(week)
  ) |>
  dplyr::group_by(player_id) |>
  dplyr::slice(1) |>
  dplyr::ungroup()

week10_rb_features <- week10_rb_candidates |>
  dplyr::inner_join(
    latest_rb_history,
    by = c("gsis_id" = "player_id"),
    suffix = c("_roster", "_history")
  )

dim(week10_rb_features)

names(week10_rb_features)

week10_rb_features |>
  dplyr::select(
    full_name,
    team_roster,
    opponent_team_roster,
    season,
    week,
    rushing_yards,
    prev_rushing_yards,
    avg_rushing_yards_3,
    avg_carries_3
  ) |>
  head(15)

dim(week10_rb_features)

week10_missing_history <- week10_rb_candidates |>
  dplyr::anti_join(
    week10_rb_features |>
      dplyr::distinct(gsis_id),
    by = "gsis_id"
  )

week10_missing_history |>
  dplyr::select(full_name, team, gsis_id)

week10_rb_features |>
  dplyr::count(season, sort = TRUE)

rb_history |>
  dplyr::select(
    player_id,
    player_name,
    season,
    week,
    rushing_yards,
    carries,
    rushing_tds,
    receiving_yards,
    receptions,
    targets
  ) |>
  dplyr::filter(
    player_id == "00-0032764",
    season == 2026
  ) |>
  dplyr::arrange(week)

rb_history |>
  dplyr::filter(
    player_id == "00-0032764",
    season == 2026,
    week == 9
  ) |>
  dplyr::select(
    week,
    prev_rushing_yards,
    prev_carries,
    avg_rushing_yards_3,
    avg_rushing_yards_5,
    avg_carries_3,
    avg_carries_5,
    sd_rushing_yards_3,
    sd_rushing_yards_5,
    sd_carries_3,
    sd_carries_5
  )

week10_player_features <- rb_history |>
  dplyr::filter(season_type == "REG") |>
  dplyr::arrange(season, week) |>
  dplyr::group_by(player_id) |>
  dplyr::summarise(
    last_season = dplyr::last(season),
    last_week = dplyr::last(week),

    prev_rushing_yards = dplyr::last(rushing_yards),
    prev_carries = dplyr::last(carries),
    prev_rushing_tds = dplyr::last(rushing_tds),
    prev_receiving_yards = dplyr::last(receiving_yards),
    prev_receptions = dplyr::last(receptions),
    prev_targets = dplyr::last(targets),

    avg_rushing_yards_3 = mean(tail(rushing_yards, 3), na.rm = TRUE),
    avg_rushing_yards_5 = mean(tail(rushing_yards, 5), na.rm = TRUE),
    avg_carries_3 = mean(tail(carries, 3), na.rm = TRUE),
    avg_carries_5 = mean(tail(carries, 5), na.rm = TRUE),

    sd_rushing_yards_3 = sd(tail(rushing_yards, 3), na.rm = TRUE),
    sd_rushing_yards_5 = sd(tail(rushing_yards, 5), na.rm = TRUE),
    sd_carries_3 = sd(tail(carries, 3), na.rm = TRUE),
    sd_carries_5 = sd(tail(carries, 5), na.rm = TRUE),

    avg_receiving_yards_3 = mean(tail(receiving_yards, 3), na.rm = TRUE),
    avg_receiving_yards_5 = mean(tail(receiving_yards, 5), na.rm = TRUE),
    avg_receptions_3 = mean(tail(receptions, 3), na.rm = TRUE),
    avg_targets_3 = mean(tail(targets, 3), na.rm = TRUE),
    avg_rushing_tds_3 = mean(tail(rushing_tds, 3), na.rm = TRUE),

    .groups = "drop"
  )

week10_player_features |>
  dplyr::filter(player_id == "00-0032764")

current_season <- 2026

latest_completed_week <- max(
  rb_history$week[
    rb_history$season == current_season &
    rb_history$season_type == "REG"
  ],
  na.rm = TRUE
)

prediction_week <- latest_completed_week + 1

latest_completed_week
prediction_week

prediction_week <- min(
  schedule_df$week[
    schedule_df$season == current_season &
    schedule_df$game_type == "REG" &
    schedule_df$week > latest_completed_week
  ],
  na.rm = TRUE
)

prediction_week

player_features <- rb_history |>
  dplyr::filter(
    season_type == "REG",
    season < current_season |
      (season == current_season & week <= latest_completed_week)
  ) |>
  dplyr::arrange(season, week) |>
  dplyr::group_by(player_id) |>
  dplyr::summarise(
    last_season = dplyr::last(season),
    last_week = dplyr::last(week),

    prev_rushing_yards = dplyr::last(rushing_yards),
    prev_carries = dplyr::last(carries),
    prev_rushing_tds = dplyr::last(rushing_tds),
    prev_receiving_yards = dplyr::last(receiving_yards),
    prev_receptions = dplyr::last(receptions),
    prev_targets = dplyr::last(targets),

    avg_rushing_yards_3 = mean(tail(rushing_yards, 3), na.rm = TRUE),
    avg_rushing_yards_5 = mean(tail(rushing_yards, 5), na.rm = TRUE),
    avg_carries_3 = mean(tail(carries, 3), na.rm = TRUE),
    avg_carries_5 = mean(tail(carries, 5), na.rm = TRUE),

    sd_rushing_yards_3 = sd(tail(rushing_yards, 3), na.rm = TRUE),
    sd_rushing_yards_5 = sd(tail(rushing_yards, 5), na.rm = TRUE),
    sd_carries_3 = sd(tail(carries, 3), na.rm = TRUE),
    sd_carries_5 = sd(tail(carries, 5), na.rm = TRUE),

    avg_receiving_yards_3 = mean(tail(receiving_yards, 3), na.rm = TRUE),
    avg_receiving_yards_5 = mean(tail(receiving_yards, 5), na.rm = TRUE),
    avg_receptions_3 = mean(tail(receptions, 3), na.rm = TRUE),
    avg_targets_3 = mean(tail(targets, 3), na.rm = TRUE),
    avg_rushing_tds_3 = mean(tail(rushing_tds, 3), na.rm = TRUE),

    .groups = "drop"
  )

dim(player_features)

names(defense_data)
dim(defense_data)

defense_features <- defense_data |>
  dplyr::filter(
    season < current_season |
      (season == current_season & week <= latest_completed_week)
  ) |>
  dplyr::arrange(season, week) |>
  dplyr::group_by(defense_team) |>
  dplyr::summarise(
    prev_rushing_yards_allowed = dplyr::last(rushing_yards_allowed),
    prev_rushing_attempts_allowed = dplyr::last(rushing_attempts_allowed),
    prev_rushing_tds_allowed = dplyr::last(rushing_tds_allowed),
    prev_rushing_epa_allowed = dplyr::last(rushing_epa_allowed),

    prev_yards_per_carry_allowed =
      dplyr::last(rushing_yards_allowed) /
      dplyr::last(rushing_attempts_allowed),

    avg_rushing_yards_allowed_3 =
      mean(tail(rushing_yards_allowed, 3), na.rm = TRUE),

    avg_rushing_attempts_allowed_3 =
      mean(tail(rushing_attempts_allowed, 3), na.rm = TRUE),

    avg_rushing_tds_allowed_3 =
      mean(tail(rushing_tds_allowed, 3), na.rm = TRUE),

    avg_rushing_yards_allowed_5 =
      mean(tail(rushing_yards_allowed, 5), na.rm = TRUE),

    avg_rushing_attempts_allowed_5 =
      mean(tail(rushing_attempts_allowed, 5), na.rm = TRUE),

    avg_rushing_tds_allowed_5 =
      mean(tail(rushing_tds_allowed, 5), na.rm = TRUE),

    .groups = "drop"
  )

dim(defense_features)
defense_features |>
  dplyr::filter(defense_team == "SF")

roster_week <- max(
  rosters_2026$week[
    rosters_2026$season == current_season &
    rosters_2026$week <= latest_completed_week
  ],
  na.rm = TRUE
)

roster_week

rb_roster_current <- rosters_2026 |>
  dplyr::filter(
    season == current_season,
    week == roster_week,
    position == "RB",
    depth_chart_position == "RB",
    status == "ACT"
  ) |>
  dplyr::select(
    full_name,
    team,
    gsis_id
  ) |>
  dplyr::distinct(gsis_id, .keep_all = TRUE)

upcoming_games <- schedule_df[
  schedule_df$season == current_season &
  schedule_df$game_type == "REG" &
  schedule_df$week == prediction_week,
]

away_matchups <- data.frame(
  team = upcoming_games$away_team,
  opponent_team = upcoming_games$home_team,
  is_home = 0
)

home_matchups <- data.frame(
  team = upcoming_games$home_team,
  opponent_team = upcoming_games$away_team,
  is_home = 1
)

upcoming_matchups <- rbind(
  away_matchups,
  home_matchups
)

upcoming_rb_candidates <- merge(
  as.data.frame(rb_roster_current),
  upcoming_matchups,
  by = "team"
)

upcoming_rb_data <- upcoming_rb_candidates |>
  dplyr::inner_join(
    player_features,
    by = c("gsis_id" = "player_id")
  ) |>
  dplyr::left_join(
    defense_features,
    by = c("opponent_team" = "defense_team")
  )

dim(upcoming_rb_data)

upcoming_rb_data |>
  dplyr::select(
    full_name,
    team,
    opponent_team,
    last_season,
    last_week,
    prev_rushing_yards,
    avg_rushing_yards_3,
    avg_rushing_yards_allowed_3
  ) |>
  head(15)

names(schedule_context)
colnames(x_all)

away_context <- data.frame(
  team = upcoming_games$away_team,
  is_home = 0,
  rest_days = upcoming_games$away_rest,
  roof = upcoming_games$roof,
  surface = upcoming_games$surface,
  temp = upcoming_games$temp,
  wind = upcoming_games$wind
)

home_context <- data.frame(
  team = upcoming_games$home_team,
  is_home = 1,
  rest_days = upcoming_games$home_rest,
  roof = upcoming_games$roof,
  surface = upcoming_games$surface,
  temp = upcoming_games$temp,
  wind = upcoming_games$wind
)

upcoming_context <- rbind(
  away_context,
  home_context
)

upcoming_rb_data <- upcoming_rb_data |>
  dplyr::left_join(
    upcoming_context,
    by = c("team", "is_home")
  )

upcoming_rb_data |>
  dplyr::select(
    full_name,
    team,
    opponent_team,
    is_home,
    rest_days,
    roof,
    surface,
    temp,
    wind
  ) |>
  head(15)

colSums(is.na(
  upcoming_rb_data[, c(
    "rest_days", "roof", "surface", "temp", "wind"
  )]
))

upcoming_rb_data <- upcoming_rb_data |>
  dplyr::mutate(
    season = current_season,
    week = prediction_week
  )

levels(predictor_data$roof)
levels(predictor_data$surface)

upcoming_predictors <- upcoming_rb_data |>
  dplyr::select(
    season,
    week,
    is_home,
    rest_days,
    roof,
    surface,
    temp,
    wind,
    prev_rushing_yards,
    prev_carries,
    prev_rushing_tds,
    prev_receiving_yards,
    prev_receptions,
    prev_targets,
    avg_rushing_yards_3,
    avg_rushing_yards_5,
    avg_carries_3,
    avg_carries_5,
    avg_receiving_yards_3,
    avg_receiving_yards_5,
    avg_receptions_3,
    avg_targets_3,
    avg_rushing_tds_3,
    sd_rushing_yards_3,
    sd_rushing_yards_5,
    sd_carries_3,
    sd_carries_5,
    prev_rushing_yards_allowed,
    prev_rushing_attempts_allowed,
    prev_rushing_tds_allowed,
    prev_rushing_epa_allowed,
    prev_yards_per_carry_allowed,
    avg_rushing_yards_allowed_3,
    avg_rushing_attempts_allowed_3,
    avg_rushing_tds_allowed_3,
    avg_rushing_yards_allowed_5,
    avg_rushing_attempts_allowed_5,
    avg_rushing_tds_allowed_5
  ) |>
  dplyr::mutate(
    roof = factor(roof, levels = levels(predictor_data$roof)),
    surface = factor(surface, levels = levels(predictor_data$surface))
  )

upcoming_frame <- model.frame(
  predictor_formula,
  data = upcoming_predictors,
  na.action = na.pass
)

x_upcoming <- model.matrix(
  predictor_formula,
  data = upcoming_frame
)

dim(x_upcoming)

identical(colnames(x_upcoming), colnames(x_all))

d_upcoming <- xgb.DMatrix(
  data = x_upcoming,
  missing = NA
)

upcoming_predictions <- predict(
  rb_xgb_production,
  d_upcoming
)

rb_predictions <- upcoming_rb_data |>
  dplyr::mutate(
    predicted_rushing_yards = upcoming_predictions
  ) |>
  dplyr::select(
    full_name,
    team,
    opponent_team,
    is_home,
    last_season,
    last_week,
    prev_rushing_yards,
    avg_rushing_yards_3,
    predicted_rushing_yards
  ) |>
  dplyr::arrange(dplyr::desc(predicted_rushing_yards))

head(rb_predictions, 20)

length(upcoming_predictions)

rb_predictions |>
  dplyr::count(last_season, last_week, sort = TRUE)

rb_predictions <- rb_predictions |>
  dplyr::mutate(
    history_status = dplyr::case_when(
      last_season < current_season ~ "Previous season",
      last_week == latest_completed_week ~ "Current week",
      latest_completed_week - last_week <= 2 ~ "Recent",
      TRUE ~ "Stale"
    )
  )

rb_predictions |>
  dplyr::filter(history_status != "Current week") |>
  dplyr::select(
    full_name, team, opponent_team,
    last_season, last_week,
    predicted_rushing_yards, history_status
  )

rb_predictions <- upcoming_rb_data |>
  dplyr::mutate(
    predicted_rushing_yards = upcoming_predictions
  ) |>
  dplyr::select(
    full_name,
    team,
    opponent_team,
    is_home,
    last_season,
    last_week,
    prev_carries,
    avg_carries_3,
    avg_carries_5,
    prev_rushing_yards,
    avg_rushing_yards_3,
    predicted_rushing_yards
  )

rb_predictions |>
  dplyr::arrange(dplyr::desc(avg_carries_3)) |>
  dplyr::select(
    full_name,
    team,
    opponent_team,
    prev_carries,
    avg_carries_3,
    avg_carries_5,
    predicted_rushing_yards
  ) |>
  head(30)

rb_predictions |>
  dplyr::filter(avg_carries_3 >= 8) |>
  dplyr::arrange(team, dplyr::desc(avg_carries_3)) |>
  dplyr::select(
    full_name,
    team,
    opponent_team,
    prev_carries,
    avg_carries_3,
    avg_carries_5,
    predicted_rushing_yards
  )

sum(rb_predictions$avg_carries_3 >= 8, na.rm = TRUE)

rb_predictions |>
  dplyr::filter(avg_carries_3 >= 8) |>
  dplyr::count(team, sort = TRUE) |>
  dplyr::filter(n > 1)

rb_predictions |>
  dplyr::filter(avg_carries_3 >= 8) |>
  dplyr::arrange(team, dplyr::desc(avg_carries_3)) |>
  dplyr::select(
    team,
    full_name,
    prev_carries,
    avg_carries_3,
    avg_carries_5,
    predicted_rushing_yards
  )

rb_predictions <- rb_predictions |>
  dplyr::left_join(
    upcoming_rb_data |>
      dplyr::select(
        full_name,
        team,
        avg_targets_3,
        avg_receptions_3
      ),
    by = c("full_name", "team")
  )

rb_predictions |>
  dplyr::arrange(dplyr::desc(avg_targets_3)) |>
  dplyr::select(
    full_name,
    team,
    prev_carries,
    avg_carries_3,
    avg_targets_3,
    avg_receptions_3,
    predicted_rushing_yards
  ) |>
  head(30)

rb_predictions_filtered <- rb_predictions |>
  dplyr::filter(
    avg_carries_3 >= 8 |
    avg_targets_3 >= 4 |
    avg_receptions_3 >= 3
  ) |>
  dplyr::arrange(team, dplyr::desc(avg_carries_3))

nrow(rb_predictions_filtered)

rb_predictions_filtered |>
  dplyr::filter(full_name == "Justice Hill")

rb_predictions_filtered |>
  dplyr::count(team, sort = TRUE)

all_teams <- sort(unique(rb_predictions$team))

included_teams <- sort(unique(rb_predictions_filtered$team))

setdiff(all_teams, included_teams)

length(unique(rb_predictions$team))
sort(unique(rb_predictions$team))

rb_weekly_predictions <- rb_predictions_filtered |>
  dplyr::mutate(
    history_status = dplyr::case_when(
      last_season < current_season ~ "Previous season",
      last_week == latest_completed_week ~ "Current week",
      latest_completed_week - last_week <= 2 ~ "Recent",
      TRUE ~ "Stale"
    ),
    predicted_rushing_yards = round(predicted_rushing_yards, 1)
  ) |>
  dplyr::arrange(
    team,
    dplyr::desc(predicted_rushing_yards)
  ) |>
  dplyr::select(
    full_name,
    team,
    opponent_team,
    is_home,
    predicted_rushing_yards,
    prev_carries,
    avg_carries_3,
    avg_targets_3,
    avg_receptions_3,
    last_season,
    last_week,
    history_status
  )

View(rb_weekly_predictions)

names(rb_modeling_data)
dim(rb_modeling_data)

players <- load_players()

names(players)

players |>
  dplyr::select(
    gsis_id,
    display_name,
    position,
    headshot
  ) |>
  head(10)

player_lookup <- players |>
  dplyr::select(
    gsis_id,
    display_name,
    position,
    headshot
  ) |>
  dplyr::distinct(gsis_id, .keep_all = TRUE)

dim(player_lookup)
head(player_lookup, 10)

names(upcoming_rb_data)

rb_weekly_predictions <- rb_weekly_predictions |>
  dplyr::left_join(
    player_lookup,
    by = c("full_name" = "display_name")
  )

upcoming_rb_data |>
  dplyr::select(gsis_id, full_name, team) |>
  head(10)

upcoming_rb_data <- upcoming_rb_data |>
  dplyr::left_join(
    player_lookup |>
      dplyr::select(gsis_id, headshot),
    by = "gsis_id"
  )

upcoming_rb_data |>
  dplyr::select(full_name, team, gsis_id, headshot) |>
  head(10)

sum(is.na(upcoming_rb_data$headshot))

rb_weekly_predictions <- rb_predictions |>
  # keep your existing filter and arrange steps
  dplyr::select(
    full_name,
    headshot,
    team,
    opponent_team,
    is_home,
    predicted_rushing_yards,
    prev_carries,
    avg_carries_3,
    avg_targets_3,
    avg_receptions_3,
    last_season,
    last_week,
    history_status
  )

rb_predictions <- rb_predictions |>
  dplyr::left_join(
    upcoming_rb_data |> dplyr::select(gsis_id, headshot),
    by = "gsis_id"
  )
