library(torch)
library(luz)
library(tidyverse)
library(torchvision)
library(nflreadr)
library(ggplot2)
library(nflplotR)

# ============================================================
# NFL GAME PREDICTION MODEL
# Training: 2020-2024
# Test:     2025
# Predict:  2026
#
# The model uses only information available BEFORE each game.
# ============================================================

# EPA = Expected Points Added
# Success = whether a play was considered successful
# YPP = Yards Per Play

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

pbp <- load_pbp(2020:2026)

games <- load_schedules(2020:2026)

glimpse(pbp)
glimpse(games)


# ------------------------------------------------------------
# 2. CREATE GAME RESULTS
# ------------------------------------------------------------

schedule <- games |>
  select(
    game_id,
    season,
    week,
    game_type,
    home_team,
    away_team,
    home_score,
    away_score
  ) |>
  mutate(
    home_win = case_when(
      home_score > away_score ~ 1,
      home_score < away_score ~ 0,
      TRUE ~ NA_real_
    )
  )

head(schedule)


# ------------------------------------------------------------
# 3. CALCULATE TEAM PERFORMANCE FOR EACH GAME
# ------------------------------------------------------------

# IMPORTANT:
# These statistics describe what happened in each game.
# We will shift them forward one game later so that they
# are NOT used to predict the same game.

team_game_stats <- pbp |>
  filter(
    season_type == "REG",
    !is.na(posteam)
  ) |>
  group_by(
    game_id,
    season,
    week,
    posteam
  ) |>
  summarise(
    epa_per_play = mean(epa, na.rm = TRUE),
    success_rate = mean(success, na.rm = TRUE),
    yards_per_play = mean(yards_gained, na.rm = TRUE),

    turnovers = sum(
      interception == 1 | fumble_lost == 1,
      na.rm = TRUE
    ),

    .groups = "drop"
  )


# ------------------------------------------------------------
# 4. CREATE PREVIOUS-GAME TEAM AVERAGES
# ------------------------------------------------------------

# Arrange each team's games chronologically.
#
# lag() makes sure the current game's statistics are NOT
# included in the information used to predict that game.

team_history <- team_game_stats |>
  arrange(posteam, season, week, game_id) |>
  group_by(posteam) |>
  mutate(
    games_before = row_number() - 1,

    previous_epa = lag(
      cummean(replace_na(epa_per_play, 0))
    ),

    previous_success = lag(
      cummean(replace_na(success_rate, 0))
    ),

    previous_ypp = lag(
      cummean(replace_na(yards_per_play, 0))
    ),

    previous_turnovers = lag(
      cummean(replace_na(turnovers, 0))
    )
  ) |>
  ungroup()


# ------------------------------------------------------------
# 5. CREATE HOME AND AWAY PRE-GAME FEATURES
# ------------------------------------------------------------

home_history <- team_history |>
  rename(
    home_team = posteam,
    home_epa = previous_epa,
    home_success = previous_success,
    home_ypp = previous_ypp,
    home_turnovers = previous_turnovers
  ) |>
  select(
    game_id,
    home_team,
    home_epa,
    home_success,
    home_ypp,
    home_turnovers
  )

away_history <- team_history |>
  rename(
    away_team = posteam,
    away_epa = previous_epa,
    away_success = previous_success,
    away_ypp = previous_ypp,
    away_turnovers = previous_turnovers
  ) |>
  select(
    game_id,
    away_team,
    away_epa,
    away_success,
    away_ypp,
    away_turnovers
  )


# ------------------------------------------------------------
# 6. CREATE MODEL DATASET
# ------------------------------------------------------------

model1 <- schedule |>
  left_join(
    home_history,
    by = c("game_id", "home_team")
  ) |>
  left_join(
    away_history,
    by = c("game_id", "away_team")
  ) |>
  mutate(
    epa_diff = home_epa - away_epa,
    success_diff = home_success - away_success,
    ypp_diff = home_ypp - away_ypp,
    turnover_diff = home_turnovers - away_turnovers
  )


# Only completed games with enough previous information
# can be used for model training/testing.

model1_complete <- model1 |>
  filter(
    !is.na(home_win),
    !is.na(home_epa),
    !is.na(away_epa),
    !is.na(home_success),
    !is.na(away_success),
    !is.na(home_ypp),
    !is.na(away_ypp),
    !is.na(home_turnovers),
    !is.na(away_turnovers)
  )

nrow(model1_complete)

glimpse(model1_complete)


# ------------------------------------------------------------
# 7. SPLIT BY SEASON
# ------------------------------------------------------------

# 2020-2024 = training
# 2025      = test

train_data <- model1_complete |>
  filter(season >= 2020, season <= 2024)

test_data <- model1_complete |>
  filter(season == 2025)

nrow(train_data)
nrow(test_data)


# ------------------------------------------------------------
# 8. CREATE TRAINING FEATURES
# ------------------------------------------------------------

feature_columns <- c(
  "home_epa",
  "away_epa",
  "home_success",
  "away_success",
  "home_ypp",
  "away_ypp",
  "home_turnovers",
  "away_turnovers",
  "epa_diff",
  "success_diff",
  "ypp_diff"
)


x_train_raw <- train_data |>
  select(all_of(feature_columns)) |>
  as.matrix()

x_test_raw <- test_data |>
  select(all_of(feature_columns)) |>
  as.matrix()


y_train <- train_data$home_win
y_test <- test_data$home_win


# ------------------------------------------------------------
# 9. SCALE USING TRAINING DATA ONLY
# ------------------------------------------------------------

x_train <- scale(x_train_raw)

train_center <- attr(x_train, "scaled:center")
train_scale <- attr(x_train, "scaled:scale")

x_test <- scale(
  x_test_raw,
  center = train_center,
  scale = train_scale
)


# ------------------------------------------------------------
# 10. CONVERT TO TORCH TENSORS
# ------------------------------------------------------------

x_train_tensor <- torch_tensor(
  x_train,
  dtype = torch_float()
)

y_train_tensor <- torch_tensor(
  y_train,
  dtype = torch_float()
)

x_test_tensor <- torch_tensor(
  x_test,
  dtype = torch_float()
)

y_test_tensor <- torch_tensor(
  y_test,
  dtype = torch_float()
)


# ------------------------------------------------------------
# 11. CREATE NEURAL NETWORK
# ------------------------------------------------------------

set.seed(1)

nfl_nn <- nn_module(

  initialize = function(input_size) {

    self$hidden <- nn_linear(
      input_size,
      20
    )

    self$dropout <- nn_dropout(
      0.5
    )

    self$activation <- nn_relu()

    self$output <- nn_linear(
      20,
      1
    )
  },

  forward = function(x) {

    x |>
      self$hidden() |>
      self$activation() |>
      self$dropout() |>
      self$output() |>
      torch_sigmoid()
  }
)


# ------------------------------------------------------------
# 12. TRAIN MODEL
# ------------------------------------------------------------

nfl_nn <- nfl_nn |>
  setup(
    loss = nn_bce_loss(),
    optimizer = optim_rmsprop
  ) |>
  set_hparams(
    input_size = ncol(x_train)
  )


fitted <- nfl_nn |>
  fit(
    data = list(
      x_train_tensor,
      y_train_tensor
    ),
    epochs = 30
  )


plot(fitted)


# ------------------------------------------------------------
# 13. TEST MODEL ON 2025
# ------------------------------------------------------------

test_probs <- predict(
  fitted,
  x_test_tensor
)

test_probs <- as.numeric(test_probs)

test_classes <- ifelse(
  test_probs > 0.5,
  1,
  0
)

test_accuracy <- mean(
  test_classes == y_test
)

cat(
  "2025 Test Accuracy:",
  round(test_accuracy, 4),
  "\n"
)


# ------------------------------------------------------------
# 14. CREATE 2025 PREDICTIONS
# ------------------------------------------------------------

test_predictions <- test_data |>
  mutate(
    home_win_prob = test_probs,
    away_win_prob = 1 - test_probs,

    predicted_winner = ifelse(
      home_win_prob > 0.5,
      home_team,
      away_team
    ),

    actual_winner = ifelse(
      home_win == 1,
      home_team,
      away_team
    ),

    correct =
      predicted_winner == actual_winner
  )


# 2025 accuracy
accuracy_2025 <- mean(
  test_predictions$correct,
  na.rm = TRUE
)

cat(
  "2025 Prediction Accuracy:",
  round(accuracy_2025, 4),
  "\n"
)


# ------------------------------------------------------------
# 15. PREPARE 2026 GAMES
# ------------------------------------------------------------

games_2026 <- games |>
  filter(
    season == 2026,
    game_type == "REG"
  ) |>
  select(
    game_id,
    season,
    week,
    game_type,
    home_team,
    away_team,
    home_score,
    away_score
  )


# ------------------------------------------------------------
# 16. CREATE 2026 PREGAME FEATURES
# ------------------------------------------------------------

season_2026 <- games_2026 |>
  left_join(
    home_history,
    by = c("game_id", "home_team")
  ) |>
  left_join(
    away_history,
    by = c("game_id", "away_team")
  ) |>
  mutate(
    epa_diff = home_epa - away_epa,
    success_diff = home_success - away_success,
    ypp_diff = home_ypp - away_ypp,
    turnover_diff = home_turnovers - away_turnovers
  )


# ------------------------------------------------------------
# 17. HANDLE TEAMS WITHOUT HISTORY
# ------------------------------------------------------------

# For 2026 teams that do not have previous-game statistics,
# use their available historical values when possible.

season_2026_model <- season_2026 |>
  filter(
    !is.na(home_epa),
    !is.na(away_epa),
    !is.na(home_success),
    !is.na(away_success),
    !is.na(home_ypp),
    !is.na(away_ypp),
    !is.na(home_turnovers),
    !is.na(away_turnovers)
  )


# ------------------------------------------------------------
# 18. SCALE 2026 DATA USING TRAINING PARAMETERS
# ------------------------------------------------------------

season_2026_x <- season_2026_model |>
  select(all_of(feature_columns)) |>
  as.matrix()

season_2026_x <- scale(
  season_2026_x,
  center = train_center,
  scale = train_scale
)

season_2026_tensor <- torch_tensor(
  season_2026_x,
  dtype = torch_float()
)


# ------------------------------------------------------------
# 19. PREDICT 2026
# ------------------------------------------------------------

season_2026_probs <- predict(
  fitted,
  season_2026_tensor
)

season_2026_probs <- as.numeric(
  season_2026_probs
)


# ------------------------------------------------------------
# 20. CREATE 2026 PREDICTION TABLE
# ------------------------------------------------------------

season_predict <- season_2026_model |>
  mutate(

    home_win_prob = season_2026_probs,

    away_win_prob =
      1 - season_2026_probs,

    predicted_winner = ifelse(
      home_win_prob > 0.5,
      home_team,
      away_team
    ),

    actual_winner = case_when(
      home_score > away_score ~ home_team,
      away_score > home_score ~ away_team,
      TRUE ~ NA_character_
    ),

    correct = case_when(
      is.na(actual_winner) ~ NA,
      TRUE ~ predicted_winner == actual_winner
    )
  )


# ------------------------------------------------------------
# 21. PRINT 2026 PREDICTIONS
# ------------------------------------------------------------

season_predict |>
  select(
    game_id,
    week,
    home_team,
    away_team,
    home_win_prob,
    away_win_prob,
    predicted_winner,
    actual_winner,
    correct
  )


# ------------------------------------------------------------
# 22. CURRENT 2026 ACCURACY
# ------------------------------------------------------------

accuracy_2026 <- mean(
  season_predict$correct,
  na.rm = TRUE
)

cat(
  "2026 Accuracy on Completed Games:",
  round(accuracy_2026, 4),
  "\n"
)


# ------------------------------------------------------------
# 23. SAVE 2026 PREDICTIONS
# ------------------------------------------------------------

write.csv(
  season_predict,
  "season_predict2.csv",
  row.names = FALSE
)


season_predict_clean <- season_predict |>
  select(
    game_id,
    week,
    home_team,
    away_team,
    home_win_prob,
    away_win_prob,
    predicted_winner,
    actual_winner,
    correct
  )


write.csv(
  season_predict_clean,
  "season_predict_clean2.csv",
  row.names = FALSE
)


# ------------------------------------------------------------
# 24. PREDICTED WINS BY TEAM
# ------------------------------------------------------------

predicted_wins <- season_predict |>
  group_by(predicted_winner) |>
  summarise(
    predicted_wins = n(),
    .groups = "drop"
  ) |>
  arrange(
    desc(predicted_wins)
  )


# ------------------------------------------------------------
# 25. PLOT PREDICTED WINS
# ------------------------------------------------------------

ggplot(
  predicted_wins,
  aes(
    x = reorder(
      predicted_winner,
      predicted_wins
    ),
    y = predicted_wins,
    fill = predicted_winner
  )
) +
  geom_col() +
  coord_flip() +
  scale_fill_nfl(type = "primary") +
  labs(
    title = "Predicted Wins by Team (2026 Season)",
    x = "Team",
    y = "Predicted Wins"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none"
  )


# ------------------------------------------------------------
# 26. ACTUAL WINS FOR COMPLETED 2026 GAMES
# ------------------------------------------------------------

actual_wins <- season_predict |>
  filter(
    !is.na(actual_winner)
  ) |>
  group_by(actual_winner) |>
  summarise(
    actual_wins = n(),
    .groups = "drop"
  )


# ------------------------------------------------------------
# 27. PREDICTED VS ACTUAL WINS
# ------------------------------------------------------------

wins_compare <- predicted_wins |>
  rename(
    team = predicted_winner
  ) |>
  left_join(
    actual_wins |>
      rename(
        team = actual_winner
      ),
    by = "team"
  ) |>
  replace_na(
    list(actual_wins = 0)
  )


ggplot(
  wins_compare,
  aes(
    x = predicted_wins,
    y = actual_wins
  )
) +
  geom_point() +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed"
  ) +
  geom_nfl_logos(
    aes(team_abbr = team),
    width = 0.06,
    position = position_jitter(
      width = 0.2,
      height = 0.2
    )
  ) +
  labs(
    title = "Predicted Wins vs Actual Wins",
    x = "Predicted Wins",
    y = "Actual Wins"
  )


# ------------------------------------------------------------
# END OF MODEL
# ------------------------------------------------------------

# 2025 Brier Score
brier_2025 <- mean(
  (test_probs - y_test)^2
)

# 2025 Log Loss
log_loss_2025 <- -mean(
  y_test * log(test_probs + 1e-15) +
    (1 - y_test) * log(1 - test_probs + 1e-15)
)

cat(
  "2025 Brier Score:",
  round(brier_2025, 4),
  "\n"
)

cat(
  "2025 Log Loss:",
  round(log_loss_2025, 4),
  "\n"
)
