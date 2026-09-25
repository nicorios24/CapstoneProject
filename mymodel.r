library(torch)
library(luz)
library(tidyverse)
library(torchvision)
library(nflreadr)
library(ggplot2)
library(nflplotR)

#epa = Expected points after play - Expected points before play
#success = success rate of plays


pbp <- load_pbp(2020:2025) #play by play from 2020-2025
games <- load_schedules(2020:2025) #games/schedules from 2020-2025

glimpse(pbp)

schedule <- games |> #will create the schedule of the games and put home_win = 1 if the home team won
  select(game_id, home_team, away_team, home_score, away_score) |> #if away team won then home_win = 0
  mutate(home_win = ifelse(home_score > away_score, 1, 0))

head(schedule)

team_stats <- pbp |>
  group_by(game_id, posteam) |>
  summarise(
    epa_per_play = mean(epa, na.rm = TRUE),
    success_rate = mean(success, na.rm = TRUE),
    yards_per_play = mean(yards_gained, na.rm = TRUE),
    turnovers = sum(interception == 1 | fumble_lost == 1, na.rm = TRUE),
    .groups = "drop"
  )

home_stats <- team_stats |>
  rename(
    home_team = posteam,
    home_epa = epa_per_play,
    home_success = success_rate,
    home_ypp = yards_per_play,
    home_turnovers = turnovers
  )

away_stats <- team_stats |>
  rename(
    away_team = posteam,
    away_epa = epa_per_play,
    away_success = success_rate,
    away_ypp = yards_per_play,
    away_turnovers = turnovers 
  )

model1 <- schedule |>
  left_join(home_stats, by = c("game_id", "home_team")) |>
  left_join(away_stats, by = c("game_id", "away_team")) |>
  select(
    game_id, home_team, away_team,
    home_epa, away_epa,
    home_success, away_success,
    home_ypp, away_ypp,
    home_turnovers, away_turnovers,
    home_win
  ) |>
  drop_na()

nrow(model1)
glimpse(model1)

model1 <- model1 |>
  mutate(
    epa_diff = home_epa - away_epa,
    success_diff = home_success - away_success,
    ypp_diff = home_ypp - away_ypp
  ) #positive = home team is better; negative = away team is better

glimpse(model1)

x <- model1 |>
  select(
    home_epa, away_epa,
    home_success, away_success,
    home_ypp, away_ypp,
    home_turnovers, away_turnovers,
    epa_diff, success_diff,
    ypp_diff
  ) |>
  scale() |>
  as.matrix()

y <- model1$home_win

x_tensor <- torch_tensor(x, dtype = torch_float())
y_tensor <- torch_tensor(y, dtype = torch_float())

set.seed(1)

train_idx <- sample(1:nrow(x), 0.8 * nrow(x))

x_train <- x_tensor[train_idx,]
y_train <- y_tensor[train_idx]

x_test <- x_tensor[-train_idx,]
y_test <- y_tensor[-train_idx]


nfl_nn <- nn_module(
  initialize = function(input_size) {
    self$hidden <- nn_linear(input_size, 20)
    self$dropout <- nn_dropout(0.5)
    self$activation <- nn_relu()
    self$output <- nn_linear(20, 1)
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

nfl_nn <- nfl_nn |>
  setup(
    loss = nn_bce_loss(),
    optimizer = optim_rmsprop
  ) |>
  set_hparams(input_size = ncol(x))

fitted <- nfl_nn |>
  fit(
    data = list(x_train, y_train),
    epochs = 30
  )

plot(fitted)

predict(fitted, x_test)

pred_probs <- predict(fitted, x_test)
pred_classes <- (pred_probs > 0.5) * 1
accuracy <- mean(as.numeric(pred_classes) == as.numeric(y_test))
accuracy


team_avg <- pbp |>
  group_by(posteam) |>
  summarise(
    epa = mean(epa, na.rm = TRUE),
    success = mean(success, na.rm = TRUE),
    ypp = mean(yards_gained, na.rm = TRUE),
    turnovers = sum(interception == 1 | fumble_lost == 1, na.rm = TRUE) / n(),
    .groups = "drop"
  )

home <- "KC"
away <- "SF"

home_stats <- team_avg |>
  filter(posteam == home)

away_stats <- team_avg |>
  filter(posteam == away)

game_features <- tibble(
  home_epa = home_stats$epa,
  away_epa = away_stats$epa,
  home_success = home_stats$success,
  away_success = away_stats$success,
  home_ypp = home_stats$ypp,
  away_ypp = away_stats$ypp,
  home_turnovers = home_stats$turnovers,
  away_turnovers = away_stats$turnovers,
  epa_diff = home_stats$epa - away_stats$epa,
  success_diff = home_stats$success - away_stats$success,
  ypp_diff = home_stats$ypp - away_stats$ypp
)

game_x <- game_features |>
  as.matrix()

game_x <- scale(
  game_x,
  center = attr(x, "scaled:center"),
  scale = attr(x, "scaled:scale")
)

game_x <- torch_tensor(game_x, dtype = torch_float())

game_prediction <- as.numeric(predict(fitted, game_x))
game_calibration <- 0.5 + (game_prediction - 0.5) * 0.5 # pulls predictions toward 0.5
cat("Win probability:", round(game_calibration, 3), "\n")
if(game_calibration > 0.5) {
  cat(home, "is predicted to WIN\n")
} else {
  cat(away, "is predicted to WIN\n")
}

# Predict all 2025 games
season_games <- games |>
  filter(season == 2025) |>
  select(game_id, home_team, away_team)

season_games <- season_games |>
  left_join(team_avg, by = c("home_team" = "posteam")) |>
  rename(
    home_epa = epa,
    home_success = success,
    home_ypp = ypp,
    home_turnovers = turnovers
  ) |>
  left_join(team_avg, by = c("away_team" = "posteam")) |>
  rename(
    away_epa = epa,
    away_success = success,
    away_ypp = ypp,
    away_turnovers = turnovers
  )

season_games <- season_games |>
  mutate(
    epa_diff = home_epa - away_epa,
    success_diff = home_success - away_success,
    ypp_diff = home_ypp - away_ypp,
    turnover_diff = home_turnovers - away_turnovers
  )

season_x <- season_games |>
  select(
    home_epa, away_epa,
    home_success, away_success,
    home_ypp, away_ypp,
    home_turnovers, away_turnovers,
    epa_diff, success_diff,
    ypp_diff  #turnover_diff
  ) |>
  as.matrix()

season_x <- sweep(season_x, 2, attr(x, "scaled:center"), "-")
season_x <- sweep(season_x, 2, attr(x, "scaled:scale"), "/")

# Convert to tensor
season_x <- torch_tensor(season_x, dtype = torch_float())

# Predict
season_preds <- as.numeric(predict(fitted, season_x))

# Calibration
season_preds_cal <- 0.5 + (season_preds - 0.5) * 0.5

# Predictions on season games
season_predict <- season_games |>
  mutate(
    home_win_prob = season_preds_cal,
    away_win_prob = 1 - season_preds_cal,
    predicted_winner = ifelse(home_win_prob > 0.5, home_team, away_team)
  )

# Print results
season_predict |>
  select(game_id, home_team, away_team, home_win_prob, away_win_prob, predicted_winner)

# Adding actual game results
season_predict <- season_predict |>
  left_join(
    games |>
      select(game_id, home_score, away_score),
    by = "game_id"
  )

season_predict <- season_predict |>
  mutate(
    actual_winner = ifelse(home_score > away_score, home_team, away_team)
  )

season_predict <- season_predict |>
  mutate(
    correct = predicted_winner == actual_winner
  )

accuracy_of_predictions <- mean(season_predict$correct, na.rm = TRUE)
accuracy_of_predictions

season_predict |>
  select(
    game_id, home_team, away_team,
    home_win_prob, away_win_prob,
    predicted_winner, actual_winner,
    correct
  ) 

  write.csv(season_predict, "season_predict.csv", row.names = FALSE)

season_predict_clean <- season_predict |>
  select(
    game_id, home_team, away_team,
    home_win_prob, away_win_prob, 
    predicted_winner, actual_winner,
    correct
  )

write.csv(season_predict_clean, "season_predict_clean.csv", row.names = FALSE)

summary(season_predict$home_score)

# Plots

# Plot 1
predicted_wins <- season_predict |>
  group_by(predicted_winner) |>
  summarise(predicted_wins = n(), .groups = "drop") |>
  arrange(desc(predicted_wins))

ggplot(predicted_wins, aes(x = reorder(predicted_winner, predicted_wins), y = predicted_wins, fill = predicted_winner)) +
  geom_col() +
  coord_flip() +
  scale_fill_nfl(type = "primary") +
  labs(
    title = "Predicted Wins by Team (2025 Season)",
    x = "Team",
    y = "Predicted Wins"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

# Plot 2
actual_wins <- season_predict |>
  group_by(actual_winner) |>
  summarise(actual_wins = n(), .groups = "drop")

wins_compare <- predicted_wins |>
  rename(team = predicted_winner) |>
  left_join(actual_wins |> rename(team = actual_winner), by = "team") |>
  replace_na(list(actual_wins = 0))

ggplot(wins_compare, aes(x = predicted_wins, y = actual_wins)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  geom_nfl_logos(aes(team_abbr = team), width = 0.06, position = position_jitter(width = 0.2, height = 0.2)) +
  labs(
    title = "Predicted Wins vs Actual Wins",
    x = "Predicted Wins",
    y = "Actual Wins"
  )

# Plot 3
team_avg |>
  arrange(desc(epa)) |>
  slice_head(n = 10) |>
  ggplot(aes(x = reorder(posteam, epa), y = epa, fill = posteam)) +
  geom_col() +
  scale_fill_nfl(type = "primary") +
  geom_nfl_logos(aes(team_abbr = posteam), width = 0.05) +
  coord_flip() +
  labs(
    title = "Top 10 Teams by EPA",
    x = "Team",
    y = "EPA per Play"
  )

# Plot 4
team_avg |>
  arrange(desc(ypp)) |>
  slice_head(n = 10) |>
  ggplot(aes(x = reorder(posteam, ypp), y = ypp, fill = posteam)) +
  geom_col() +
  scale_fill_nfl(type = "primary") +
  geom_nfl_logos(aes(team_abbr = posteam), width = 0.05) +
  coord_flip() +
  labs(
    title = "Top 10 Teams by YPP",
    x = "Team",
    y = "Yards per Play"
  )

# Plot 5
team_avg |>
  arrange(desc(turnovers)) |>
  slice_head(n = 10) |>
  ggplot(aes(x = reorder(posteam, turnovers), y = turnovers, fill = posteam)) +
  geom_col() +
  scale_fill_nfl(type = "primary") +
  geom_nfl_logos(aes(team_abbr = posteam), width = 0.05) +
  coord_flip() +
  labs(
    title = "Top 10 Teams by Turnovers",
    x = "Team",
    y = "Turnovers"
  )
