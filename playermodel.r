library(torch)
library(luz)
library(tidyverse)
library(nflreadr)
library(dplyr)
library(slider)

player_stats <- load_player_stats(2020:2026)

glimpse(player_stats)

player_data <- player_stats |>
  filter(season_type == "REG")

rb_data <- player_data |>
  filter(position == "RB") |>
  select(
    player_id,
    player_display_name,
    season,
    week,
    game_id,
    team,
    opponent_team,
    carries,
    rushing_yards,
    rushing_tds,
    rushing_fumbles_lost,
    receptions,
    targets,
    receiving_yards,
    receiving_tds,
    receiving_fumbles_lost
  )

glimpse(rb_data)

rb_model_data <- rb_data |>
  arrange(player_id, season, week) |>
  group_by(player_id) |>
  mutate(
    prev_carries = lag(carries),
    prev_rushing_yards = lag(rushing_yards),
    prev_rushing_tds = lag(rushing_tds),
    prev_rushing_fumbles = lag(rushing_fumbles_lost),
    prev_receptions = lag(receptions),
    prev_targets = lag(targets),
    prev_receiving_yards = lag(receiving_yards),
    prev_receiving_tds = lag(receiving_tds),
    prev_receiving_fumbles = lag(receiving_fumbles_lost)
  ) |>
  ungroup()

names(rb_model_data)

glimpse(rb_model_data)

rb_model_data_og <- rb_model_data

rb_model_data <- rb_model_data |>
  arrange(player_id, season, week) |>
  group_by(player_id) |>
  mutate(
    avg_carries_last_3 = slide_dbl(carries, mean, .before = 2, .after = -1, .complete = FALSE),
    avg_rushing_yards_last_3 = slide_dbl(rushing_yards, mean, .before = 2, .after = -1, .complete = FALSE),
    avg_targets_last_3 = slide_dbl(targets, mean, .before = 2, .after = -1, .complete = FALSE),
    avg_receptions_last_3 = slide_dbl(receptions, mean, .before = 2, .after = -1, .complete = FALSE),
    avg_carries_last_5 = slide_dbl(carries, mean, .before = 4, .after = -1, .complete = FALSE),
    avg_rushing_yards_last_5 = slide_dbl(rushing_yards, mean, .before = 4, .after = -1, .complete = FALSE),
    avg_targets_last_5 = slide_dbl(targets, mean, .before = 4, .after = -1, .complete = FALSE),
    avg_receptions_last_5 = slide_dbl(receptions, mean, .before = 4, .after = -1, .complete = FALSE)
  ) |>
  ungroup()

glimpse(rb_model_data)

rb_model_data |>
  filter(player_display_name == "Frank Gore", season == 2020) |>
  select(
    player_display_name,
    season,
    week,
    carries,
    rushing_yards,
    avg_carries_last_3,
    avg_rushing_yards_last_3,
    avg_carries_last_5,
    avg_rushing_yards_last_5
  ) |>
  head(10)

prev_season_stats <- rb_model_data |>
  group_by(player_id, season) |>
  summarise(
    prev_season_avg_carries = mean(carries, na.rm = TRUE),
    prev_season_avg_rushing_yards = mean(rushing_yards, na.rm = TRUE),
    prev_season_avg_targets = mean(targets, na.rm = TRUE),
    prev_season_avg_receptions = mean(receptions, na.rm = TRUE),
    prev_season_total_rushing_yards = sum(rushing_yards, na.rm = TRUE),
    prev_season_total_rushing_tds = sum(rushing_tds, na.rm = TRUE),
    prev_season_games = n(),
    .groups = "drop"
  ) |>
  mutate(
    season = season + 1
  )

rb_model_data <- rb_model_data %>%
  left_join(
    prev_season_stats,
    by = c("player_id", "season")
  )

rb_model_data %>%
  filter(
    player_display_name == "Ezekiel Elliott",
    season == 2021,
    week == 1
  ) %>%
  select(
    player_display_name,
    season,
    week,
    carries,
    rushing_yards,
    prev_carries,
    prev_rushing_yards,
    avg_carries_last_3,
    avg_rushing_yards_last_3,
    prev_season_avg_carries,
    prev_season_avg_rushing_yards,
    prev_season_total_rushing_yards,
    prev_season_total_rushing_tds,
    prev_season_games
  )

rb_model_data %>%
  filter(
    player_display_name == "Ezekiel Elliott",
    season == 2021,
    week == 1
  ) %>%
  select(
    player_display_name,
    season,
    week,
    carries,
    rushing_yards,
    prev_carries,
    prev_rushing_yards,
    avg_carries_last_3,
    avg_rushing_yards_last_3,
    avg_carries_last_5,
    avg_rushing_yards_last_5,
    prev_season_avg_carries,
    prev_season_avg_rushing_yards,
    prev_season_avg_targets,
    prev_season_avg_receptions,
    prev_season_total_rushing_yards,
    prev_season_total_rushing_tds,
    prev_season_games
  )
