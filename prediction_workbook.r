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
library(writexl)
library(nflreadr)
library(tidyverse)

list.files(pattern = "\\.csv$", ignore.case = TRUE)

# Import RB predictions
rb_data <- read.csv("rb_2026_predictions.csv")

# Import WR predictions
wr_data <- read.csv("WR_2026_Neural_Network_Predictions.csv")

# Import TE predictions
te_data <- read.csv("TE_2026_Neural_Network_Predictions.csv")

# Import QB predictions
qb_data <- read.csv("QB_2026_Neural_Network_Predictions.csv")

# Import team predictions
team_data <- read.csv("season_predict_clean.csv")

# Check the imported data
dim(rb_data)
dim(wr_data)
dim(te_data)
dim(qb_data)
dim(team_data)

# Check column names
names(rb_data)
names(wr_data)
names(te_data)
names(qb_data)
names(team_data)

rb_data <- rb_data |>
  dplyr::rename(
    player_display_name = player,
    opponent_team = opponent
  )

te_data <- te_data |>
  dplyr::rename(
    actual_targets = targets,
    actual_receptions = receptions,
    actual_receiving_yards = receiving_yards,
    actual_receiving_tds = receiving_tds
  )

wr_data <- wr_data |>
  dplyr::rename(
    predicted_receiving_yards = predicted_yards,
    predicted_td_probability = td_probability
  )

names(qb_data) <- tolower(names(qb_data))

names(rb_data)
names(wr_data)
names(te_data)
names(qb_data)

all_player_predictions <- bind_rows(
  QB = qb_data,
  RB = rb_data,
  WR = wr_data,
  TE = te_data,
  .id = "position"
)

dim(all_player_predictions)

table(all_player_predictions$position)

all_player_predictions %>%
  count(week, position)

player_predictions <- all_player_predictions %>%
  arrange(week, position, player_display_name)

team_predictions <- team_data %>%
  arrange(week)

write_xlsx(
  list(
    "Player Predictions" = player_predictions,
    "Team Predictions" = team_predictions
  ),
  "NFL_2026_Predictions.xlsx"
)

file.exists("NFL_2026_Predictions.xlsx")

names(qb_data)
names(rb_data)
names(wr_data)
names(te_data)

player_info <- load_players() %>%
  select(
    player_id = gsis_id,
    player_display_name = display_name,
    headshot_url = headshot
  ) %>%
  distinct()

names(player_info)

head(player_info)

player_info %>%
  count(player_display_name) %>%
  filter(n > 1)
