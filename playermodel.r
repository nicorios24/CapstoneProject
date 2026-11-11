
# ==========================================
# NFL RUNNING BACK RUSHING YARDS PREDICTION
# ==========================================


# 1. LOAD LIBRARIES

library(torch)
library(luz)
library(tidyverse)
library(nflreadr)
library(dplyr)
library(slider)


# 2. LOAD NFL PLAYER STATISTICS

player_stats <- load_player_stats(2020:2026)

glimpse(player_stats)


# 3. FILTER REGULAR SEASON DATA

player_data <- player_stats |>
  filter(season_type == "REG")


# 4. EXTRACT RUNNING BACK STATISTICS

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
  ) |>
  arrange(player_id, season, week)

glimpse(rb_data)


# 5. HELPER FUNCTION FOR HISTORICAL AVERAGES

mean_or_na <- function(x) {
  if (length(x) == 0 || all(is.na(x))) {
    return(NA_real_)
  }

  mean(x, na.rm = TRUE)
}


# 6. CREATE PREVIOUS-GAME AND ROLLING FEATURES

rb_model_data <- rb_data |>
  arrange(player_id, season, week) |>
  group_by(player_id) |>
  mutate(

    # Number of previous games in our dataset
    games_played_before = row_number() - 1,

    # Previous-game statistics
    prev_carries = lag(carries),
    prev_rushing_yards = lag(rushing_yards),
    prev_rushing_tds = lag(rushing_tds),
    prev_rushing_fumbles = lag(rushing_fumbles_lost),
    prev_receptions = lag(receptions),
    prev_targets = lag(targets),
    prev_receiving_yards = lag(receiving_yards),
    prev_receiving_tds = lag(receiving_tds),
    prev_receiving_fumbles = lag(receiving_fumbles_lost),

    # Previous 3-game averages
    avg_carries_last_3 = slide_dbl(
      carries, mean_or_na,
      .before = 2, .after = -1
    ),

    avg_rushing_yards_last_3 = slide_dbl(
      rushing_yards, mean_or_na,
      .before = 2, .after = -1
    ),

    avg_targets_last_3 = slide_dbl(
      targets, mean_or_na,
      .before = 2, .after = -1
    ),

    avg_receptions_last_3 = slide_dbl(
      receptions, mean_or_na,
      .before = 2, .after = -1
    ),

    # Previous 5-game averages
    avg_carries_last_5 = slide_dbl(
      carries, mean_or_na,
      .before = 4, .after = -1
    ),

    avg_rushing_yards_last_5 = slide_dbl(
      rushing_yards, mean_or_na,
      .before = 4, .after = -1
    ),

    avg_targets_last_5 = slide_dbl(
      targets, mean_or_na,
      .before = 4, .after = -1
    ),

    avg_receptions_last_5 = slide_dbl(
      receptions, mean_or_na,
      .before = 4, .after = -1
    )

  ) |>
  ungroup()

glimpse(rb_model_data)


# 7. CALCULATE PREVIOUS-SEASON STATISTICS

prev_season_stats <- rb_data |>
  group_by(player_id, season) |>
  summarise(

    prev_season_avg_carries = mean(carries, na.rm = TRUE),

    prev_season_avg_rushing_yards = mean(
      rushing_yards, na.rm = TRUE
    ),

    prev_season_avg_targets = mean(targets, na.rm = TRUE),

    prev_season_avg_receptions = mean(
      receptions, na.rm = TRUE
    ),

    prev_season_total_rushing_yards = sum(
      rushing_yards, na.rm = TRUE
    ),

    prev_season_total_rushing_tds = sum(
      rushing_tds, na.rm = TRUE
    ),

    prev_season_games = n(),

    .groups = "drop"

  ) |>
  mutate(
    season = season + 1
  )


# 8. JOIN PREVIOUS-SEASON STATISTICS

rb_model_data <- rb_model_data |>
  left_join(
    prev_season_stats,
    by = c("player_id", "season")
  )


# 9. CALCULATE HISTORICAL PERFORMANCE AGAINST OPPONENT

rb_model_data <- rb_model_data |>
  arrange(player_id, season, week) |>
  group_by(player_id, opponent_team) |>
  mutate(

    # Previous games against this opponent
    games_vs_opponent = row_number() - 1,

    # Average rushing yards in previous matchups
    avg_rushing_yards_vs_opponent = lag(
      cummean(rushing_yards)
    )

  ) |>
  ungroup()


# 10. INSPECT THE COMPLETED DATASET

glimpse(rb_model_data)

names(rb_model_data)


# 11. CHECK ROOKIES AND PLAYERS WITH LIMITED HISTORY

rb_model_data |>
  filter(games_played_before < 3) |>
  select(
    player_display_name,
    season,
    week,
    games_played_before,
    prev_rushing_yards,
    avg_rushing_yards_last_3
  ) |>
  head(15)


# 12. CHECK OPPONENT HISTORY

rb_model_data |>
  select(
    player_display_name,
    season,
    week,
    opponent_team,
    games_vs_opponent,
    avg_rushing_yards_vs_opponent
  ) |>
  filter(games_vs_opponent > 0) |>
  head(15)

rb_model_data <- rb_model_data |>
  rename(games_in_dataset_before = games_played_before)

# 13. CHECK PREVIOUS-SEASON STATISTICS

rb_model_data |>
  filter(
    player_display_name == "Ezekiel Elliott",
    season == 2021,
    week == 1
  ) |>
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


# ==========================================
# 14. THREE-GAME BASELINE EVALUATION
# ==========================================

# Keep this baseline consistent with our
# original 21.3 MAE evaluation.

rb_baseline_data <- rb_data |>
  arrange(player_id, season, week) |>
  group_by(player_id) |>
  mutate(
    predicted_rushing_yards = slide_dbl(
      rushing_yards,
      mean,
      .before = 2,
      .after = -1,
      .complete = TRUE
    )
  ) |>
  ungroup()


# 15. EVALUATE ON 2025 TEST SEASON

rb_baseline_test <- rb_baseline_data |>
  filter(
    season == 2025,
    !is.na(predicted_rushing_yards),
    !is.na(rushing_yards)
  ) |>
  mutate(
    absolute_error = abs(
      rushing_yards - predicted_rushing_yards
    )
  )


# 16. CALCULATE MAE

baseline_mae <- rb_baseline_test |>
  summarise(
    MAE = mean(absolute_error),
    games_evaluated = n()
  )

baseline_mae


# 17. INSPECT LARGEST PREDICTION ERRORS

rb_baseline_test |>
  select(
    player_display_name,
    season,
    week,
    rushing_yards,
    predicted_rushing_yards,
    absolute_error
  ) |>
  arrange(desc(absolute_error)) |>
  head(10)

rb_features <- c(
  "games_in_dataset_before",
  "prev_rushing_yards",
  "prev_carries",
  "avg_rushing_yards_last_3",
  "avg_carries_last_3",
  "avg_rushing_yards_last_5",
  "prev_season_avg_rushing_yards",
  "games_vs_opponent",
  "avg_rushing_yards_vs_opponent"
)

rb_model_ready <- rb_model_data |>
  select(
    player_id,
    player_display_name,
    season,
    week,
    rushing_yards,
    all_of(rb_features)
  ) |>
  filter(!is.na(rushing_yards)) |>
  mutate(
    across(all_of(rb_features), as.numeric)
  )

rb_train_raw <- rb_model_ready |>
  filter(season >= 2020, season <= 2024)

rb_test_raw <- rb_model_ready |>
  filter(season == 2025)

# Record which historical features are missing
rb_train <- rb_train_raw |>
  mutate(
    across(
      all_of(rb_features),
      ~ as.integer(is.na(.x)),
      .names = "{.col}_missing"
    )
  )

rb_test <- rb_test_raw |>
  mutate(
    across(
      all_of(rb_features),
      ~ as.integer(is.na(.x)),
      .names = "{.col}_missing"
    )
  )

# Calculate medians using training data only
training_medians <- rb_train_raw |>
  summarise(
    across(
      all_of(rb_features),
      function(x) {
        value <- median(x, na.rm = TRUE)

        if (is.finite(value)) value else 0
      }
    )
  )

# Fill missing values in training data
rb_train <- rb_train |>
  mutate(
    across(
      all_of(rb_features),
      function(x) {
        ifelse(
          is.na(x),
          training_medians[[cur_column()]],
          x
        )
      }
    )
  )

# Use the same training medians for testing data
rb_test <- rb_test |>
  mutate(
    across(
      all_of(rb_features),
      function(x) {
        ifelse(
          is.na(x),
          training_medians[[cur_column()]],
          x
        )
      }
    )
  )

# Number of records
nrow(rb_train)
nrow(rb_test)

# Confirm the seasons
table(rb_train$season)
table(rb_test$season)

# Check missing values in the predictors
colSums(is.na(rb_train[rb_features]))
colSums(is.na(rb_test[rb_features]))

# Look at the prepared training data
head(rb_train)

# Include the missing-value indicators
missing_features <- grep(
  "_missing$",
  names(rb_train),
  value = TRUE
)

model_features <- c(rb_features, missing_features)

# Split training data into training and validation
rb_train_core <- rb_train |>
  filter(season >= 2020, season <= 2023)

rb_validation <- rb_train |>
  filter(season == 2024)

# Create feature matrices
x_train <- as.matrix(
  rb_train_core |> select(all_of(model_features))
)

x_validation <- as.matrix(
  rb_validation |> select(all_of(model_features))
)

x_test <- as.matrix(
  rb_test |> select(all_of(model_features))
)

# Calculate scaling values using training data only
feature_means <- colMeans(x_train)
feature_sds <- apply(x_train, 2, sd)

# Avoid division by zero
feature_sds[feature_sds == 0] <- 1

# Standardize all three datasets using training values
x_train <- scale(
  x_train,
  center = feature_means,
  scale = feature_sds
)

x_validation <- scale(
  x_validation,
  center = feature_means,
  scale = feature_sds
)

x_test <- scale(
  x_test,
  center = feature_means,
  scale = feature_sds
)

y_train <- rb_train_core$rushing_yards
y_validation <- rb_validation$rushing_yards

target_mean <- mean(y_train)
target_sd <- sd(y_train)

y_train_scaled <- (y_train - target_mean) / target_sd
y_validation_scaled <- (y_validation - target_mean) / target_sd

x_train_tensor <- torch_tensor(
  x_train,
  dtype = torch_float()
)

y_train_tensor <- torch_tensor(
  matrix(y_train_scaled, ncol = 1),
  dtype = torch_float()
)

x_validation_tensor <- torch_tensor(
  x_validation,
  dtype = torch_float()
)

y_validation_tensor <- torch_tensor(
  matrix(y_validation_scaled, ncol = 1),
  dtype = torch_float()
)

x_test_tensor <- torch_tensor(
  x_test,
  dtype = torch_float()
)

train_dataset <- tensor_dataset(
  x_train_tensor,
  y_train_tensor
)

validation_dataset <- tensor_dataset(
  x_validation_tensor,
  y_validation_tensor
)

train_dl <- dataloader(
  train_dataset,
  batch_size = 64,
  shuffle = TRUE
)

validation_dl <- dataloader(
  validation_dataset,
  batch_size = 64
)

rb_model <- nn_module(
  initialize = function(input_size) {

    self$fc1 <- nn_linear(input_size, 32)
    self$fc2 <- nn_linear(32, 16)
    self$dropout <- nn_dropout(p = 0.2)
    self$output <- nn_linear(16, 1)

  },

  forward = function(x) {

    x <- self$fc1(x)
    x <- nnf_relu(x)

    x <- self$dropout(x)

    x <- self$fc2(x)
    x <- nnf_relu(x)

    x <- self$output(x)

    x
  }
)

fitted_rb_model <- rb_model |>
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam
  ) |>
  set_hparams(
    input_size = ncol(x_train)
  ) |>
  fit(
    train_dl,
    epochs = 50,
    valid_data = validation_dl
  )

fitted_rb_model_21 <- rb_model |>
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam
  ) |>
  set_hparams(
    input_size = ncol(x_train)
  ) |>
  fit(
    train_dl,
    epochs = 21,
    valid_data = validation_dl
  )

# Generate predictions
test_predictions_scaled <- predict(
  fitted_rb_model_21,
  x_test_tensor
)

# Convert predictions back into rushing yards
predicted_yards <- as.numeric(test_predictions_scaled) *
  target_sd + target_mean

# Create a results dataset
rb_predictions <- rb_test |>
  select(
    player_id,
    player_display_name,
    season,
    week,
    rushing_yards
  ) |>
  mutate(
    predicted_rushing_yards = predicted_yards
  )

rb_model_results <- rb_predictions |>
  mutate(
    absolute_error = abs(
      rushing_yards - predicted_rushing_yards
    ),
    squared_error = (
      rushing_yards - predicted_rushing_yards
    )^2
  )

rb_model_metrics <- rb_model_results |>
  summarise(
    MAE = mean(absolute_error),
    RMSE = sqrt(mean(squared_error)),
    games_evaluated = n()
  )

rb_model_metrics

rb_comparison <- rb_predictions |>
  inner_join(
    rb_baseline_test |>
      select(
        player_id,
        season,
        week,
        baseline_prediction = predicted_rushing_yards
      ),
    by = c("player_id", "season", "week")
  ) |>
  mutate(
    model_error = abs(
      rushing_yards - predicted_rushing_yards
    ),
    baseline_error = abs(
      rushing_yards - baseline_prediction
    )
  )

rb_comparison_metrics <- rb_comparison |>
  summarise(
    Neural_Network_MAE = mean(model_error),
    Baseline_MAE = mean(baseline_error),
    Games_Evaluated = n(),
    Improvement = mean(baseline_error) - mean(model_error)
  )

rb_comparison_metrics

# Load NFL schedules for 2020-2026
nfl_schedules <- load_schedules(2020:2026)

# Keep regular-season games and relevant weather information
weather_data <- nfl_schedules |>
  filter(game_type == "REG") |>
  select(
    game_id,
    season,
    week,
    roof,
    temp,
    wind,
    surface,
    stadium
  ) |>
  distinct(game_id, .keep_all = TRUE)

weather_data <- weather_data |>
  mutate(
    temp = as.numeric(temp),
    wind = as.numeric(wind),

    is_indoor = case_when(
      roof %in% c("dome", "closed") ~ 1,
      roof %in% c("outdoors", "open") ~ 0,
      TRUE ~ NA_real_
    ),

    roof_open = case_when(
      roof == "open" ~ 1,
      roof %in% c("dome", "outdoors", "closed") ~ 0,
      TRUE ~ NA_real_
    ),

    is_grass = case_when(
      str_detect(str_to_lower(surface), "grass") ~ 1,
      !is.na(surface) ~ 0,
      TRUE ~ NA_real_
    )
  )

rb_model_data_v2 <- rb_model_data |>
  left_join(
    weather_data |>
      select(
        game_id,
        temp,
        wind,
        is_indoor,
        roof_open,
        is_grass
      ),
    by = "game_id"
  )

# Confirm that the number of player-game records did not change
nrow(rb_model_data)
nrow(rb_model_data_v2)

# Inspect weather information
rb_model_data_v2 |>
  select(
    player_display_name,
    season,
    week,
    temp,
    wind,
    is_indoor,
    roof_open,
    is_grass
  ) |>
  head(10)

# Count missing values
colSums(
  is.na(
    rb_model_data_v2[
      c("temp", "wind", "is_indoor", "roof_open", "is_grass")
    ]
  )
)

# Check stadium roof categories
weather_data |> count(roof)

# Check available playing surfaces
weather_data |> count(surface)

rb_model_data_v2 |>
  group_by(is_indoor) |>
  summarise(
    total_games = n(),
    missing_temp = sum(is.na(temp)),
    missing_wind = sum(is.na(wind)),
    missing_percentage = mean(is.na(temp)) * 100
  )

rb_model_data_v2 |>
  group_by(season) |>
  summarise(
    total_games = n(),
    missing_temp = sum(is.na(temp)),
    missing_wind = sum(is.na(wind))
  )

# Investigate outdoor games with missing weather
rb_model_data_v2 |>
  filter(is_indoor == 0, is.na(temp)) |>
  group_by(season) |>
  summarise(
    missing_games = n(),
    stadiums = n_distinct(game_id),
    .groups = "drop"
  )

# Check which stadiums have missing outdoor weather
rb_model_data_v2 <- rb_model_data_v2 |>
  left_join(
    weather_data |>
      select(game_id, stadium),
    by = "game_id"
  )

rb_model_data_v2 |>
  filter(is_indoor == 0, is.na(temp)) |>
  count(stadium, sort = TRUE)

rb_model_data_v2 |>
  filter(is_indoor == 0, is.na(temp)) |>
  count(season, sort = TRUE)

library(httr2)
library(jsonlite)

weather_url <- "https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=33.9535&longitude=-118.3392&start_date=2025-10-21&end_date=2025-10-22&hourly=temperature_2m,relative_humidity_2m,precipitation,rain,snowfall,wind_speed_10m,wind_gusts_10m&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=America%2FLos_Angeles"

response <- request(weather_url) |>
  req_perform()

weather_result <- response |>
  resp_body_json(simplifyVector = TRUE)

weather_test <- as_tibble(weather_result$hourly)

head(weather_test)

nfl_schedules |>
  filter(
    season == 2025,
    stadium == "SoFi Stadium"
  ) |>
  select(
    game_id,
    season,
    week,
    gameday,
    gametime,
    home_team,
    away_team,
    stadium
  ) |>
  head()

# Weather for an actual NFL game
game_weather_response <- request(
  "https://historical-forecast-api.open-meteo.com/v1/forecast"
) |>
  req_url_query(
    latitude = 33.9535,
    longitude = -118.3392,
    start_date = "2025-09-05",
    end_date = "2025-09-05",
    hourly = paste(
      "temperature_2m",
      "relative_humidity_2m",
      "precipitation",
      "rain",
      "snowfall",
      "wind_speed_10m",
      "wind_gusts_10m",
      sep = ","
    ),
    temperature_unit = "fahrenheit",
    wind_speed_unit = "mph",
    timezone = "America/Los_Angeles"
  ) |>
  req_perform()

game_weather_result <- resp_body_json(
  game_weather_response,
  simplifyVector = TRUE
)

game_weather <- as_tibble(game_weather_result$hourly)

# Display weather at kickoff
game_weather |>
  filter(time == "2025-09-05T20:00")

names(weather_result)

weather_result$hourly_units

weather_result$elevation

game_weather

# Historical weather at SoFi Stadium
actual_weather_response <- request(
  "https://archive-api.open-meteo.com/v1/archive"
) |>
  req_url_query(
    latitude = 33.9535,
    longitude = -118.3392,
    start_date = "2025-09-05",
    end_date = "2025-09-05",
    hourly = paste(
      "temperature_2m",
      "relative_humidity_2m",
      "precipitation",
      "rain",
      "snowfall",
      "wind_speed_10m",
      "wind_gusts_10m",
      sep = ","
    ),
    temperature_unit = "fahrenheit",
    wind_speed_unit = "mph",
    timezone = "America/Los_Angeles"
  ) |>
  req_perform()

actual_weather_result <- resp_body_json(
  actual_weather_response,
  simplifyVector = TRUE
)

actual_weather <- as_tibble(actual_weather_result$hourly)

actual_weather |>
  filter(time == "2025-09-05T20:00")


# Get unique regular-season NFL games from 2020-2026
games <- nfl_schedules |>
  filter(game_type == "REG") |>
  select(
    game_id,
    season,
    week,
    gameday,
    gametime,
    home_team,
    away_team,
    stadium,
    roof,
    surface
  ) |>
  distinct(game_id, .keep_all = TRUE)

# Check the number of games
nrow(games)

# Check for missing game information
games |>
  summarise(
    total_games = n(),
    missing_game_id = sum(is.na(game_id)),
    missing_date = sum(is.na(gameday)),
    missing_time = sum(is.na(gametime)),
    missing_stadium = sum(is.na(stadium))
  )

# List every stadium in the schedule
games |>
  count(stadium, sort = TRUE)

games |>
  count(stadium, sort = TRUE) |>
  print(n = Inf)


# Download an existing NFL stadium coordinate dataset
stadium_coordinates_source <- read_csv(
  "https://raw.githubusercontent.com/ThompsonJamesBliss/WeatherData/master/data/stadium_coordinates.csv"
)

# Keep the unique stadium coordinates
stadium_coordinates_source <- stadium_coordinates_source |>
  select(
    StadiumName,
    Longitude,
    Latitude
  ) |>
  distinct()

# Only include games that actually appear in our RB dataset
games_for_weather <- games |>
  semi_join(
    rb_model_data |> distinct(game_id),
    by = "game_id"
  )

# Get stadium names used by our actual player-game records
stadiums_needed <- games_for_weather |>
  distinct(stadium) |>
  arrange(stadium)

# Find stadium names that do not match the coordinate dataset
unmatched_stadiums <- stadiums_needed |>
  anti_join(
    stadium_coordinates_source,
    by = c("stadium" = "StadiumName")
  )

# Display the results
nrow(games_for_weather)

unmatched_stadiums |>
  print(n = Inf)


# Manually add stadiums missing from the downloaded dataset
manual_stadium_coordinates <- tribble(
  ~StadiumName, ~Latitude, ~Longitude,

  # United States
  "Acrisure Stadium", 40.4468, -80.0158,
  "Caesars Superdome", 29.9511, -90.0812,
  "EverBank Stadium", 30.3239, -81.6373,
  "GEHA Field at Arrowhead Stadium", 39.0489, -94.4839,
  "Highmark Stadium", 42.7738, -78.7870,
  "Huntington Bank Field", 41.5061, -81.6995,
  "Lumen Field", 47.5952, -122.3316,
  "Northwest Stadium", 38.9076, -76.8645,
  "Paycor Stadium", 39.0954, -84.5160,
  "Reliant Stadium", 29.6847, -95.4107,
  "TIAA Bank Stadium", 30.3239, -81.6373,

  # International stadiums
  "Allianz Arena", 48.2188, 11.6247,
  "Arena Corinthians", -23.5455, -46.4731,
  "Azteca Stadium", 19.3029, -99.1505,
  "Deutsche Bank Park", 50.0686, 8.6455,
  "Maracana Stadium", -22.9121, -43.2302,
  "Melbourne Cricket Ground", -37.8200, 144.9834,
  "Stade de France", 48.9244, 2.3601,
  "Tottenham Stadium", 51.6043, -0.0662
)

# Combine downloaded and manually entered coordinates
stadium_coordinates <- bind_rows(
  stadium_coordinates_source,
  manual_stadium_coordinates
) |>
  distinct(StadiumName, .keep_all = TRUE)

# Attach coordinates to each game
games_with_coordinates <- games_for_weather |>
  left_join(
    stadium_coordinates,
    by = c("stadium" = "StadiumName")
  )

# Check for stadiums still missing coordinates
games_with_coordinates |>
  filter(is.na(Latitude) | is.na(Longitude)) |>
  distinct(stadium) |>
  print(n = Inf)

# Check the number of games and unique stadiums
games_with_coordinates |>
  summarise(
    total_games = n(),
    unique_stadiums = n_distinct(stadium),
    missing_latitude = sum(is.na(Latitude)),
    missing_longitude = sum(is.na(Longitude))
  )

games_with_coordinates |>
  select(
    game_id,
    gameday,
    gametime,
    stadium,
    Latitude,
    Longitude
  ) |>
  head(10)

str(games_with_coordinates$gameday)
str(games_with_coordinates$gametime)


# Convert scheduled kickoff times from Eastern Time to UTC
games_with_coordinates <- games_with_coordinates |>
  mutate(
    kickoff_eastern = as.POSIXct(
      paste(gameday, gametime),
      format = "%Y-%m-%d %H:%M",
      tz = "America/New_York"
    ),

    # Convert kickoff to UTC
    kickoff_utc = as.POSIXct(
      format(
        kickoff_eastern,
        tz = "UTC",
        format = "%Y-%m-%d %H:%M:%S"
      ),
      format = "%Y-%m-%d %H:%M:%S",
      tz = "UTC"
    ),

    # Find the nearest hourly weather observation
    weather_hour_utc = as.POSIXct(
      floor(as.numeric(kickoff_utc) / 3600) * 3600,
      origin = "1970-01-01",
      tz = "UTC"
    ),

    # Format the timestamp for Open-Meteo
    weather_timestamp_utc = format(
      weather_hour_utc,
      tz = "UTC",
      format = "%Y-%m-%dT%H:%M"
    )
  )

# Inspect the converted kickoff times
games_with_coordinates |>
  select(
    game_id,
    gameday,
    gametime,
    stadium,
    kickoff_utc,
    weather_timestamp_utc
  ) |>
  head(10)


# Get all 2025 games
games_2025 <- games_with_coordinates |>
  filter(season == 2025)

# Get unique stadium locations used in 2025
weather_locations_2025 <- games_2025 |>
  distinct(Latitude, Longitude) |>
  arrange(Latitude, Longitude)

# Prepare the API request
weather_response_2025 <- request(
  "https://archive-api.open-meteo.com/v1/archive"
) |>
  req_url_query(
    latitude = paste(weather_locations_2025$Latitude, collapse = ","),
    longitude = paste(weather_locations_2025$Longitude, collapse = ","),
    start_date = min(games_2025$gameday),
    end_date = as.character(as.Date(max(games_2025$gameday)) + 1),
    hourly = paste(
      "temperature_2m",
      "relative_humidity_2m",
      "precipitation",
      "rain",
      "snowfall",
      "wind_speed_10m",
      "wind_gusts_10m",
      sep = ","
    ),
    temperature_unit = "fahrenheit",
    wind_speed_unit = "mph",
    precipitation_unit = "mm",
    timezone = "UTC"
  ) |>
  req_perform()

# Read the API response
weather_result_2025 <- resp_body_json(
  weather_response_2025,
  simplifyVector = TRUE
)


# Convert the nested hourly weather data into one table
weather_data_2025 <- purrr::map_dfr(
  seq_len(nrow(weather_result_2025)),
  function(i) {

    tibble(
      Latitude = weather_locations_2025$Latitude[i],
      Longitude = weather_locations_2025$Longitude[i],

      weather_timestamp_utc =
        weather_result_2025$hourly$time[[i]],

      temperature_2m =
        weather_result_2025$hourly$temperature_2m[[i]],

      relative_humidity_2m =
        weather_result_2025$hourly$relative_humidity_2m[[i]],

      precipitation =
        weather_result_2025$hourly$precipitation[[i]],

      rain =
        weather_result_2025$hourly$rain[[i]],

      snowfall =
        weather_result_2025$hourly$snowfall[[i]],

      wind_speed_10m =
        weather_result_2025$hourly$wind_speed_10m[[i]],

      wind_gusts_10m =
        weather_result_2025$hourly$wind_gusts_10m[[i]]
    )
  }
)

# Inspect the resulting weather table
weather_data_2025

# Check its dimensions
dim(weather_data_2025)

# Check missing weather values
weather_data_2025 |>
  summarise(
    across(
      c(
        temperature_2m,
        relative_humidity_2m,
        precipitation,
        rain,
        snowfall,
        wind_speed_10m,
        wind_gusts_10m
      ),
      ~ sum(is.na(.x))
    )
  )

# Match each 2025 game with weather at its stadium and kickoff hour
games_2025_with_weather <- games_2025 |>
  left_join(
    weather_data_2025,
    by = c(
      "Latitude",
      "Longitude",
      "weather_timestamp_utc"
    )
  )

# Check the resulting data
dim(games_2025_with_weather)

# Check for missing weather values after joining
games_2025_with_weather |>
  summarise(
    total_games = n(),
    across(
      c(
        temperature_2m,
        relative_humidity_2m,
        precipitation,
        rain,
        snowfall,
        wind_speed_10m,
        wind_gusts_10m
      ),
      ~ sum(is.na(.x))
    )
  )

games_2025_with_weather |>
  select(
    game_id,
    gameday,
    gametime,
    stadium,
    weather_timestamp_utc,
    temperature_2m,
    relative_humidity_2m,
    precipitation,
    wind_speed_10m,
    wind_gusts_10m
  ) |>
  head(10)

dim(games_2025)
dim(games_2025_with_weather)

games_2025_with_weather |>
  summarise(
    total_games = n(),
    across(
      c(
        temperature_2m,
        relative_humidity_2m,
        precipitation,
        rain,
        snowfall,
        wind_speed_10m,
        wind_gusts_10m
      ),
      ~ sum(is.na(.x))
    )
  )

# Function to retrieve historical weather for one NFL season
get_weather_for_season <- function(season_number) {

  # Get games for the selected season
  season_games <- games_with_coordinates |>
    filter(season == season_number)

  # Get unique stadium locations
  season_locations <- season_games |>
    distinct(Latitude, Longitude) |>
    arrange(Latitude, Longitude)

  # Request historical weather from Open-Meteo
  weather_response <- request(
    "https://archive-api.open-meteo.com/v1/archive"
  ) |>
    req_url_query(
      latitude = paste(season_locations$Latitude, collapse = ","),
      longitude = paste(season_locations$Longitude, collapse = ","),
      start_date = min(season_games$gameday),
      end_date = as.character(
        as.Date(max(season_games$gameday)) + 1
      ),
      hourly = paste(
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "snowfall",
        "wind_speed_10m",
        "wind_gusts_10m",
        sep = ","
      ),
      temperature_unit = "fahrenheit",
      wind_speed_unit = "mph",
      precipitation_unit = "mm",
      timezone = "UTC"
    ) |>
    req_perform()

  # Convert API response to R data
  weather_result <- resp_body_json(
    weather_response,
    simplifyVector = TRUE
  )

  # Verify that all requested locations were returned
  if (nrow(weather_result) != nrow(season_locations)) {
    stop("Location count mismatch for season ", season_number)
  }

  # Flatten hourly weather data into one table
  season_weather <- purrr::map_dfr(
    seq_len(nrow(weather_result)),
    function(i) {

      tibble(
        Latitude = season_locations$Latitude[i],
        Longitude = season_locations$Longitude[i],

        weather_timestamp_utc =
          weather_result$hourly$time[[i]],

        temperature_2m =
          weather_result$hourly$temperature_2m[[i]],

        relative_humidity_2m =
          weather_result$hourly$relative_humidity_2m[[i]],

        precipitation =
          weather_result$hourly$precipitation[[i]],

        rain =
          weather_result$hourly$rain[[i]],

        snowfall =
          weather_result$hourly$snowfall[[i]],

        wind_speed_10m =
          weather_result$hourly$wind_speed_10m[[i]],

        wind_gusts_10m =
          weather_result$hourly$wind_gusts_10m[[i]]
      )
    }
  )

  # Return the weather data
  season_weather
}

weather_data_2024 <- get_weather_for_season(2024)

dim(weather_data_2024)

head(weather_data_2024)

weather_data_2023 <- get_weather_for_season(2023)

dim(weather_data_2023)

head(weather_data_2023)

weather_data_2022 <- get_weather_for_season(2022)

dim(weather_data_2022)

head(weather_data_2022)

weather_data_2021 <- get_weather_for_season(2021)

dim(weather_data_2021)

head(weather_data_2021)

weather_data_2020 <- get_weather_for_season(2020)

dim(weather_data_2020)

head(weather_data_2020)

games_2026 <- games_with_coordinates |>
  filter(season == 2026)

games_2026 |>
  summarise(
    total_games = n(),
    first_game = min(gameday),
    last_game = max(gameday),
    unique_stadiums = n_distinct(stadium)
  )

weather_data_2026 <- get_weather_for_season(2026)

dim(weather_data_2026)

head(weather_data_2026)

# Combine weather data from all seven seasons
weather_data_all <- bind_rows(
  weather_data_2020,
  weather_data_2021,
  weather_data_2022,
  weather_data_2023,
  weather_data_2024,
  weather_data_2025,
  weather_data_2026
)

# Check dimensions
dim(weather_data_all)

# Check for duplicate location and timestamp combinations
weather_data_all |>
  count(Latitude, Longitude, weather_timestamp_utc) |>
  filter(n > 1)

# Check missing weather values
weather_data_all |>
  summarise(
    across(
      c(
        temperature_2m,
        relative_humidity_2m,
        precipitation,
        rain,
        snowfall,
        wind_speed_10m,
        wind_gusts_10m
      ),
      ~ sum(is.na(.x))
    )
  )

# Join historical weather to all NFL games
games_with_weather <- games_with_coordinates |>
  left_join(
    weather_data_all,
    by = c(
      "Latitude",
      "Longitude",
      "weather_timestamp_utc"
    )
  )

# Check dimensions
dim(games_with_coordinates)
dim(games_with_weather)

# Check missing weather values by season
games_with_weather |>
  group_by(season) |>
  summarise(
    total_games = n(),
    missing_temperature = sum(is.na(temperature_2m)),
    missing_humidity = sum(is.na(relative_humidity_2m)),
    missing_precipitation = sum(is.na(precipitation)),
    missing_wind = sum(is.na(wind_speed_10m)),
    .groups = "drop"
  )


# Add detailed historical weather to running back records
rb_model_data_v2_updated <- rb_model_data_v2 |>
  select(-temp, -wind) |>
  left_join(
    games_with_weather |>
      select(
        game_id,
        temperature_2m,
        relative_humidity_2m,
        precipitation,
        rain,
        snowfall,
        wind_speed_10m,
        wind_gusts_10m
      ),
    by = "game_id"
  )

# Check dimensions
dim(rb_model_data_v2)
dim(rb_model_data_v2_updated)

# Check missing weather values
rb_model_data_v2_updated |>
  summarise(
    across(
      c(
        temperature_2m,
        relative_humidity_2m,
        precipitation,
        rain,
        snowfall,
        wind_speed_10m,
        wind_gusts_10m
      ),
      ~ sum(is.na(.x))
    )
  )

# Version 2 features: original predictors plus detailed weather
rb_features_v2 <- c(
  rb_features,
  "temperature_2m",
  "relative_humidity_2m",
  "precipitation",
  "rain",
  "snowfall",
  "wind_speed_10m",
  "wind_gusts_10m"
)

# Prepare Version 2 modeling dataset
rb_model_ready_v2 <- rb_model_data_v2_updated |>
  select(
    player_id,
    player_display_name,
    season,
    week,
    rushing_yards,
    all_of(rb_features_v2)
  ) |>
  filter(!is.na(rushing_yards)) |>
  mutate(
    across(all_of(rb_features_v2), as.numeric)
  )

# Check dimensions
dim(rb_model_ready_v2)

# Check the predictors
rb_features_v2

# Check missing values in the new predictors
rb_model_ready_v2 |>
  summarise(
    across(
      all_of(rb_features_v2),
      ~ sum(is.na(.x))
    )
  )

# Split Version 2 data by season
rb_train_raw_v2 <- rb_model_ready_v2 |>
  filter(season %in% 2020:2024)

rb_test_raw_v2 <- rb_model_ready_v2 |>
  filter(season == 2025)

# Check missing values in each dataset
rb_train_raw_v2 |>
  summarise(
    across(
      all_of(rb_features_v2),
      ~ sum(is.na(.x))
    )
  )

rb_test_raw_v2 |>
  summarise(
    across(
      all_of(rb_features_v2),
      ~ sum(is.na(.x))
    )
  )

rb_model_ready_v2 |>
  group_by(season) |>
  summarise(
    across(
      all_of(rb_features_v2),
      ~ sum(is.na(.x))
    ),
    .groups = "drop"
  )

# Check Version 2 dataset dimensions
dim(rb_model_ready_v2)

# Check how many records we have per season
rb_model_ready_v2 |>
  count(season)

# Check missing values in each feature
rb_model_ready_v2 |>
  summarise(
    across(
      all_of(rb_features_v2),
      ~ sum(is.na(.))
    )
  )

# Review the Version 2 features
rb_features_v2

rb_model_ready_v2 |>
  summarise(
    across(
      all_of(rb_features_v2),
      ~ sum(is.na(.x))
    )
  ) |>
  pivot_longer(
    everything(),
    names_to = "feature",
    values_to = "missing"
  ) |>
  arrange(desc(missing))


# Version 2 dataset splits

rb_train_raw_v2 <- rb_model_ready_v2 |>
  filter(season %in% 2020:2023)

rb_validation_raw_v2 <- rb_model_ready_v2 |>
  filter(season == 2024)

rb_test_raw_v2 <- rb_model_ready_v2 |>
  filter(season == 2025)

# Check dataset sizes

dim(rb_train_raw_v2)
dim(rb_validation_raw_v2)
dim(rb_test_raw_v2)

# Create missing-value indicators

rb_train_v2 <- rb_train_raw_v2 |>
  mutate(
    across(
      all_of(rb_features_v2),
      ~ as.integer(is.na(.x)),
      .names = "{.col}_missing"
    )
  )

rb_validation_v2 <- rb_validation_raw_v2 |>
  mutate(
    across(
      all_of(rb_features_v2),
      ~ as.integer(is.na(.x)),
      .names = "{.col}_missing"
    )
  )

rb_test_v2 <- rb_test_raw_v2 |>
  mutate(
    across(
      all_of(rb_features_v2),
      ~ as.integer(is.na(.x)),
      .names = "{.col}_missing"
    )
  )

# Calculate medians using training data only

rb_medians_v2 <- rb_train_v2 |>
  summarise(
    across(
      all_of(rb_features_v2),
      ~ median(.x, na.rm = TRUE)
    )
  )

# Replace missing values using training medians

rb_impute_v2 <- function(data) {
  data |>
    mutate(
      across(
        all_of(rb_features_v2),
        ~ replace(
          .x,
          is.na(.x),
          rb_medians_v2[[cur_column()]]
        )
      )
    )
}

rb_train_v2 <- rb_impute_v2(rb_train_v2)
rb_validation_v2 <- rb_impute_v2(rb_validation_v2)
rb_test_v2 <- rb_impute_v2(rb_test_v2)

# Check missing values in predictors

rb_train_v2 |>
  summarise(across(all_of(rb_features_v2), ~ sum(is.na(.x))))

rb_validation_v2 |>
  summarise(across(all_of(rb_features_v2), ~ sum(is.na(.x))))

rb_test_v2 |>
  summarise(across(all_of(rb_features_v2), ~ sum(is.na(.x))))

# Combine original features and missing indicators

rb_missing_features_v2 <- paste0(rb_features_v2, "_missing")

rb_model_features_v2 <- c(
  rb_features_v2,
  rb_missing_features_v2
)

# Calculate means and standard deviations from training data

rb_x_means_v2 <- sapply(
  rb_train_v2[, rb_model_features_v2],
  mean
)

rb_x_sds_v2 <- sapply(
  rb_train_v2[, rb_model_features_v2],
  sd
)

# Prevent division by zero

rb_x_sds_v2[rb_x_sds_v2 == 0] <- 1

# Scale training, validation, and testing predictors

rb_scale_x_v2 <- function(data) {
  scale(
    as.matrix(data[, rb_model_features_v2]),
    center = rb_x_means_v2,
    scale = rb_x_sds_v2
  )
}

x_train_v2 <- rb_scale_x_v2(rb_train_v2)
x_validation_v2 <- rb_scale_x_v2(rb_validation_v2)
x_test_v2 <- rb_scale_x_v2(rb_test_v2)

# Calculate target mean and standard deviation from training data

rb_y_mean_v2 <- mean(rb_train_v2$rushing_yards)
rb_y_sd_v2 <- sd(rb_train_v2$rushing_yards)

# Standardize target values

y_train_v2 <- (rb_train_v2$rushing_yards - rb_y_mean_v2) / rb_y_sd_v2

y_validation_v2 <- (rb_validation_v2$rushing_yards - rb_y_mean_v2) / rb_y_sd_v2

y_test_v2 <- (rb_test_v2$rushing_yards - rb_y_mean_v2) / rb_y_sd_v2

dim(x_train_v2)
dim(x_validation_v2)
dim(x_test_v2)

length(y_train_v2)
length(y_validation_v2)
length(y_test_v2)

sum(is.na(x_train_v2))
sum(is.na(x_validation_v2))
sum(is.na(x_test_v2))

# Version 2 RB neural network

rb_model_v2 <- nn_module(
  initialize = function(input_size) {

    self$fc1 <- nn_linear(input_size, 32)
    self$fc2 <- nn_linear(32, 16)
    self$dropout <- nn_dropout(p = 0.2)
    self$output <- nn_linear(16, 1)

  },

  forward = function(x) {

    x <- self$fc1(x)
    x <- nnf_relu(x)

    x <- self$dropout(x)

    x <- self$fc2(x)
    x <- nnf_relu(x)

    x <- self$output(x)

    x
  }
)

# Convert Version 2 data into tensors

x_train_tensor_v2 <- torch_tensor(
  x_train_v2,
  dtype = torch_float()
)

y_train_tensor_v2 <- torch_tensor(
  y_train_v2,
  dtype = torch_float()
)

x_validation_tensor_v2 <- torch_tensor(
  x_validation_v2,
  dtype = torch_float()
)

y_validation_tensor_v2 <- torch_tensor(
  y_validation_v2,
  dtype = torch_float()
)

x_test_tensor_v2 <- torch_tensor(
  x_test_v2,
  dtype = torch_float()
)

# Create datasets

train_dataset_v2 <- tensor_dataset(
  x_train_tensor_v2,
  y_train_tensor_v2
)

validation_dataset_v2 <- tensor_dataset(
  x_validation_tensor_v2,
  y_validation_tensor_v2
)

# Create data loaders

train_dl_v2 <- dataloader(
  train_dataset_v2,
  batch_size = 64,
  shuffle = TRUE
)

validation_dl_v2 <- dataloader(
  validation_dataset_v2,
  batch_size = 64
)

fitted_rb_model_v2 <- rb_model_v2 |>
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam
  ) |>
  set_hparams(
    input_size = ncol(x_train_v2)
  ) |>
  fit(
    train_dl_v2,
    epochs = 50,
    valid_data = validation_dl_v2
  )

fitted_rb_model_v2_21 <- rb_model_v2 |>
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam
  ) |>
  set_hparams(
    input_size = ncol(x_train_v2)
  ) |>
  fit(
    train_dl_v2,
    epochs = 21,
    valid_data = validation_dl_v2
  )

test_predictions_scaled_v2 <- predict(
  fitted_rb_model_v2_21,
  x_test_tensor_v2
)

# Inspect the warnings
warnings()

# Check training tensor dimensions
dim(x_train_tensor_v2)
dim(y_train_tensor_v2)

# Check validation tensor dimensions
dim(x_validation_tensor_v2)
dim(y_validation_tensor_v2)

# Check prediction dimensions
dim(test_predictions_scaled_v2)

# Create target tensors with two dimensions

y_train_tensor_v2 <- torch_tensor(
  y_train_v2,
  dtype = torch_float()
) |>
  torch_unsqueeze(dim = 2)

y_validation_tensor_v2 <- torch_tensor(
  y_validation_v2,
  dtype = torch_float()
) |>
  torch_unsqueeze(dim = 2)

train_dataset_v2 <- tensor_dataset(
  x_train_tensor_v2,
  y_train_tensor_v2
)

validation_dataset_v2 <- tensor_dataset(
  x_validation_tensor_v2,
  y_validation_tensor_v2
)

train_dl_v2 <- dataloader(
  train_dataset_v2,
  batch_size = 64,
  shuffle = TRUE
)

validation_dl_v2 <- dataloader(
  validation_dataset_v2,
  batch_size = 64
)

dim(y_train_tensor_v2)
dim(y_validation_tensor_v2)

fitted_rb_model_v2 <- rb_model_v2 |>
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam
  ) |>
  set_hparams(
    input_size = ncol(x_train_v2)
  ) |>
  fit(
    train_dl_v2,
    epochs = 50,
    valid_data = validation_dl_v2
  )

fitted_rb_model_v2_21 <- rb_model_v2 |>
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam
  ) |>
  set_hparams(
    input_size = ncol(x_train_v2)
  ) |>
  fit(
    train_dl_v2,
    epochs = 21,
    valid_data = validation_dl_v2
  )

test_predictions_scaled_v2 <- predict(
  fitted_rb_model_v2_21,
  x_test_tensor_v2
)

test_predictions_v2 <- as.numeric(test_predictions_scaled_v2) *
  rb_y_sd_v2 + rb_y_mean_v2

fitted_rb_model_v2_7 <- rb_model_v2 |>
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam
  ) |>
  set_hparams(
    input_size = ncol(x_train_v2)
  ) |>
  fit(
    train_dl_v2,
    epochs = 7,
    valid_data = validation_dl_v2
  )

test_predictions_scaled_v2 <- predict(
  fitted_rb_model_v2_7,
  x_test_tensor_v2
)

test_predictions_v2 <- as.numeric(test_predictions_scaled_v2) *
  rb_y_sd_v2 + rb_y_mean_v2

rb_test_results_v2 <- rb_test_raw_v2 |>
  mutate(
    predicted_rushing_yards = round(test_predictions_v2, 1),
    actual_rushing_yards = rushing_yards,
    prediction_error = round(
      actual_rushing_yards - predicted_rushing_yards, 1
    )
  ) |>
  select(
    player_display_name,
    season,
    week,
    actual_rushing_yards,
    predicted_rushing_yards,
    prediction_error
  )

View(rb_test_results_v2)

rb_test_results_v2 |>
  summarise(
    MAE = mean(abs(actual_rushing_yards - predicted_rushing_yards)),
    RMSE = sqrt(mean((actual_rushing_yards - predicted_rushing_yards)^2))
  )

fitted_rb_model_v2_25 <- rb_model_v2 |>
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam
  ) |>
  set_hparams(input_size = ncol(x_train_v2)) |>
  fit(
    train_dl_v2,
    epochs = 25,
    valid_data = validation_dl_v2
  )

fitted_rb_model_v2_16 <- rb_model_v2 |>
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam
  ) |>
  set_hparams(input_size = ncol(x_train_v2)) |>
  fit(
    train_dl_v2,
    epochs = 16,
    valid_data = validation_dl_v2
  )

fitted_rb_model_v2_14 <- rb_model_v2 |>
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam
  ) |>
  set_hparams(input_size = ncol(x_train_v2)) |>
  fit(
    train_dl_v2,
    epochs = 14,
    valid_data = validation_dl_v2
  )

test_predictions_scaled_v2_16 <- predict(
  fitted_rb_model_v2_16,
  x_test_tensor_v2
)

test_predictions_v2_16 <- as.numeric(test_predictions_scaled_v2_16) *
  rb_y_sd_v2 + rb_y_mean_v2

rb_test_results_v2_16 <- rb_test_raw_v2 |>
  mutate(
    predicted_rushing_yards = round(test_predictions_v2_16, 1),
    actual_rushing_yards = rushing_yards,
    prediction_error = round(
      actual_rushing_yards - predicted_rushing_yards, 1
    )
  ) |>
  select(
    player_display_name,
    season,
    week,
    actual_rushing_yards,
    predicted_rushing_yards,
    prediction_error
  )

rb_test_results_v2_16 |>
  summarise(
    MAE = mean(abs(prediction_error)),
    RMSE = sqrt(mean(prediction_error^2))
  )

# RB MODEL VERSION 3: OPPONENT DEFENSIVE FEATURES

pbp_defense_raw_v3 <- load_pbp(2020:2026)

names(pbp_defense_raw_v3)

# Calculate rushing defense statistics for each game

defense_game_stats_v3 <- pbp_defense_raw_v3 |>
  filter(
    season_type == "REG",
    !is.na(defteam),
    rush_attempt == 1,
    qb_kneel != 1
  ) |>
  group_by(
    game_id,
    season,
    week,
    game_date,
    defteam
  ) |>
  summarise(
    rush_yards_allowed = sum(yards_gained, na.rm = TRUE),
    rush_attempts_allowed = n(),
    rush_tds_allowed = sum(rush_touchdown, na.rm = TRUE),
    avg_yards_per_carry_allowed = rush_yards_allowed / rush_attempts_allowed,
    avg_rush_epa_allowed = mean(epa, na.rm = TRUE),
    .groups = "drop"
  )

dim(defense_game_stats_v3)

head(defense_game_stats_v3, 10)

summary(defense_game_stats_v3)

defense_game_stats_v3 |>
  filter(rush_yards_allowed < 0)

pbp_defense_raw_v3 |>
  filter(
    season_type == "REG",
    rush_attempt == 1,
    defteam %in% (
      defense_game_stats_v3 |>
        filter(rush_yards_allowed < 0) |>
        pull(defteam)
    )
  ) |>
  select(
    game_id,
    posteam,
    defteam,
    desc,
    yards_gained,
    qb_kneel,
    rush_attempt
  ) |>
  filter(yards_gained < 0)

# Calculate previous five-game defensive averages

defense_recent_stats_v3 <- defense_game_stats_v3 |>
  arrange(defteam, game_date, season, week) |>
  group_by(defteam) |>
  mutate(
    opp_avg_rush_yards_allowed_last5 = slider::slide_dbl(
      lag(rush_yards_allowed),
      ~ mean(.x, na.rm = TRUE),
      .before = 4
    ),
    opp_avg_rush_attempts_allowed_last5 = slider::slide_dbl(
      lag(rush_attempts_allowed),
      ~ mean(.x, na.rm = TRUE),
      .before = 4
    ),
    opp_avg_yards_per_carry_allowed_last5 = slider::slide_dbl(
      lag(avg_yards_per_carry_allowed),
      ~ mean(.x, na.rm = TRUE),
      .before = 4
    ),
    opp_avg_rush_tds_allowed_last5 = slider::slide_dbl(
      lag(rush_tds_allowed),
      ~ mean(.x, na.rm = TRUE),
      .before = 4
    ),
    opp_avg_rush_epa_allowed_last5 = slider::slide_dbl(
      lag(avg_rush_epa_allowed),
      ~ mean(.x, na.rm = TRUE),
      .before = 4
    )
  ) |>
  ungroup()

dim(defense_recent_stats_v3)

head(defense_recent_stats_v3, 15)

summary(
  defense_recent_stats_v3 |>
    select(contains("last5"))
)

rb_model_data_v3 <- rb_model_data_v2_updated |>
  left_join(
    defense_recent_stats_v3 |>
      select(
        game_id,
        defteam,
        opp_avg_rush_yards_allowed_last5,
        opp_avg_rush_attempts_allowed_last5,
        opp_avg_yards_per_carry_allowed_last5,
        opp_avg_rush_tds_allowed_last5,
        opp_avg_rush_epa_allowed_last5
      ),
    by = c(
      "game_id",
      "opponent_team" = "defteam"
    )
  )

dim(rb_model_data_v2_updated)
dim(rb_model_data_v3)

names(rb_model_data_v3)

rb_model_data_v3 |>
  select(
    opp_avg_rush_yards_allowed_last5,
    opp_avg_rush_attempts_allowed_last5,
    opp_avg_yards_per_carry_allowed_last5,
    opp_avg_rush_tds_allowed_last5,
    opp_avg_rush_epa_allowed_last5
  ) |>
  summary()

rb_features_v3 <- c(
  rb_features_v2,
  "opp_avg_rush_yards_allowed_last5",
  "opp_avg_rush_attempts_allowed_last5",
  "opp_avg_yards_per_carry_allowed_last5",
  "opp_avg_rush_tds_allowed_last5",
  "opp_avg_rush_epa_allowed_last5"
)

rb_model_ready_v3 <- rb_model_data_v3 |>
  select(
    player_id,
    player_display_name,
    season,
    week,
    rushing_yards,
    all_of(rb_features_v3)
  ) |>
  filter(!is.na(rushing_yards)) |>
  mutate(across(all_of(rb_features_v3), as.numeric))

dim(rb_model_ready_v3)

rb_model_ready_v3 |>
  summarise(
    across(
      all_of(rb_features_v3),
      ~ sum(is.na(.x))
    )
  )

rb_train_raw_v3 <- rb_model_ready_v3 |>
  filter(season %in% 2020:2023)

rb_validation_raw_v3 <- rb_model_ready_v3 |>
  filter(season == 2024)

rb_test_raw_v3 <- rb_model_ready_v3 |>
  filter(season == 2025)

dim(rb_train_raw_v3)
dim(rb_validation_raw_v3)
dim(rb_test_raw_v3)

rb_train_v3 <- rb_train_raw_v3 |>
  mutate(across(
    all_of(rb_features_v3),
    ~ as.integer(is.na(.x)),
    .names = "{.col}_missing"
  ))

rb_validation_v3 <- rb_validation_raw_v3 |>
  mutate(across(
    all_of(rb_features_v3),
    ~ as.integer(is.na(.x)),
    .names = "{.col}_missing"
  ))

rb_test_v3 <- rb_test_raw_v3 |>
  mutate(across(
    all_of(rb_features_v3),
    ~ as.integer(is.na(.x)),
    .names = "{.col}_missing"
  ))

rb_medians_v3 <- rb_train_v3 |>
  summarise(across(
    all_of(rb_features_v3),
    ~ median(.x, na.rm = TRUE)
  ))

rb_impute_v3 <- function(data) {
  data |>
    mutate(across(
      all_of(rb_features_v3),
      ~ replace(
        .x,
        is.na(.x),
        rb_medians_v3[[cur_column()]]
      )
    ))
}

rb_train_v3 <- rb_impute_v3(rb_train_v3)
rb_validation_v3 <- rb_impute_v3(rb_validation_v3)
rb_test_v3 <- rb_impute_v3(rb_test_v3)

sum(is.na(rb_train_v3[, rb_features_v3]))
sum(is.na(rb_validation_v3[, rb_features_v3]))
sum(is.na(rb_test_v3[, rb_features_v3]))

rb_missing_features_v3 <- paste0(rb_features_v3, "_missing")

rb_model_features_v3 <- c(
  rb_features_v3,
  rb_missing_features_v3
)

length(rb_features_v3)
length(rb_model_features_v3)

rb_x_means_v3 <- sapply(
  rb_train_v3[, rb_model_features_v3],
  mean
)

rb_x_sds_v3 <- sapply(
  rb_train_v3[, rb_model_features_v3],
  sd
)

rb_x_sds_v3[rb_x_sds_v3 == 0] <- 1

rb_scale_x_v3 <- function(data) {
  scale(
    as.matrix(data[, rb_model_features_v3]),
    center = rb_x_means_v3,
    scale = rb_x_sds_v3
  )
}

x_train_v3 <- rb_scale_x_v3(rb_train_v3)
x_validation_v3 <- rb_scale_x_v3(rb_validation_v3)
x_test_v3 <- rb_scale_x_v3(rb_test_v3)

rb_y_mean_v3 <- mean(rb_train_v3$rushing_yards)
rb_y_sd_v3 <- sd(rb_train_v3$rushing_yards)

y_train_v3 <- (
  rb_train_v3$rushing_yards - rb_y_mean_v3
) / rb_y_sd_v3

y_validation_v3 <- (
  rb_validation_v3$rushing_yards - rb_y_mean_v3
) / rb_y_sd_v3

y_test_v3 <- (
  rb_test_v3$rushing_yards - rb_y_mean_v3
) / rb_y_sd_v3

dim(x_train_v3)
dim(x_validation_v3)
dim(x_test_v3)

length(y_train_v3)
length(y_validation_v3)
length(y_test_v3)

x_train_tensor_v3 <- torch_tensor(
  x_train_v3,
  dtype = torch_float()
)

x_validation_tensor_v3 <- torch_tensor(
  x_validation_v3,
  dtype = torch_float()
)

x_test_tensor_v3 <- torch_tensor(
  x_test_v3,
  dtype = torch_float()
)

y_train_tensor_v3 <- torch_tensor(
  matrix(y_train_v3, ncol = 1),
  dtype = torch_float()
)

y_validation_tensor_v3 <- torch_tensor(
  matrix(y_validation_v3, ncol = 1),
  dtype = torch_float()
)

y_test_tensor_v3 <- torch_tensor(
  matrix(y_test_v3, ncol = 1),
  dtype = torch_float()
)

train_dataset_v3 <- tensor_dataset(
  x_train_tensor_v3,
  y_train_tensor_v3
)

validation_dataset_v3 <- tensor_dataset(
  x_validation_tensor_v3,
  y_validation_tensor_v3
)

train_dl_v3 <- dataloader(
  train_dataset_v3,
  batch_size = 64,
  shuffle = TRUE
)

validation_dl_v3 <- dataloader(
  validation_dataset_v3,
  batch_size = 64
)

rb_model_v3 <- nn_module(
  initialize = function(input_size) {
    self$fc1 <- nn_linear(input_size, 32)
    self$fc2 <- nn_linear(32, 16)
    self$dropout <- nn_dropout(p = 0.2)
    self$output <- nn_linear(16, 1)
  },

  forward = function(x) {
    x <- self$fc1(x)
    x <- nnf_relu(x)
    x <- self$dropout(x)
    x <- self$fc2(x)
    x <- nnf_relu(x)
    x <- self$output(x)
    x
  }
)

x_train_tensor_v3$size()
y_train_tensor_v3$size()

x_validation_tensor_v3$size()
y_validation_tensor_v3$size()

x_test_tensor_v3$size()
y_test_tensor_v3$size()

fitted_rb_model_v3 <- rb_model_v3(input_size = 42)

optimizer_v3 <- optim_adam(
  fitted_rb_model_v3$parameters,
  lr = 0.001
)

for (epoch in 1:50) {

  fitted_rb_model_v3$train()
  train_loss <- 0

  coro::loop(for (batch in train_dl_v3) {
    optimizer_v3$zero_grad()

    predictions <- fitted_rb_model_v3(batch[[1]])
    loss <- nnf_mse_loss(predictions, batch[[2]])

    loss$backward()
    optimizer_v3$step()

    train_loss <- train_loss + loss$item()
  })

  fitted_rb_model_v3$eval()
  validation_loss <- 0

  with_no_grad({
    coro::loop(for (batch in validation_dl_v3) {
      predictions <- fitted_rb_model_v3(batch[[1]])
      loss <- nnf_mse_loss(predictions, batch[[2]])

      validation_loss <- validation_loss + loss$item()
    })
  })

  train_loss <- train_loss / length(train_dl_v3)
  validation_loss <- validation_loss / length(validation_dl_v3)

  cat(
    "Epoch:", epoch,
    "| Training Loss:", round(train_loss, 4),
    "| Validation Loss:", round(validation_loss, 4),
    "\n"
  )
}

fitted_rb_model_v3 <- rb_model_v3(input_size = 42)

optimizer_v3 <- optim_adam(
  fitted_rb_model_v3$parameters,
  lr = 0.001
)

best_validation_loss_v3 <- Inf
best_epoch_v3 <- 0
best_state_v3 <- NULL

for (epoch in 1:50) {

  fitted_rb_model_v3$train()
  train_loss <- 0

  coro::loop(for (batch in train_dl_v3) {
    optimizer_v3$zero_grad()

    predictions <- fitted_rb_model_v3(batch[[1]])
    loss <- nnf_mse_loss(predictions, batch[[2]])

    loss$backward()
    optimizer_v3$step()

    train_loss <- train_loss + loss$item()
  })

  fitted_rb_model_v3$eval()
  validation_loss <- 0

  with_no_grad({
    coro::loop(for (batch in validation_dl_v3) {
      predictions <- fitted_rb_model_v3(batch[[1]])
      loss <- nnf_mse_loss(predictions, batch[[2]])

      validation_loss <- validation_loss + loss$item()
    })
  })

  train_loss <- train_loss / length(train_dl_v3)
  validation_loss <- validation_loss / length(validation_dl_v3)

  if (validation_loss < best_validation_loss_v3) {
    best_validation_loss_v3 <- validation_loss
    best_epoch_v3 <- epoch

    best_state_v3 <- lapply(
      fitted_rb_model_v3$state_dict(),
      function(x) x$clone()
    )
  }

  cat(
    "Epoch:", epoch,
    "| Training Loss:", round(train_loss, 4),
    "| Validation Loss:", round(validation_loss, 4),
    "| Best Epoch:", best_epoch_v3,
    "| Best Validation Loss:", round(best_validation_loss_v3, 4),
    "\n"
  )
}

fitted_rb_model_v3$load_state_dict(best_state_v3)

best_epoch_v3
best_validation_loss_v3

fitted_rb_model_v3$eval()

predictions_v3 <- with_no_grad({
  fitted_rb_model_v3(x_test_tensor_v3)
})

predicted_rushing_yards_v3 <- as.numeric(
  as_array(predictions_v3)
) * rb_y_sd_v3 + rb_y_mean_v3

rb_test_results_v3 <- rb_test_v3 |>
  select(
    player_id,
    player_display_name,
    season,
    week,
    rushing_yards
  ) |>
  mutate(
    predicted_rushing_yards = predicted_rushing_yards_v3,
    prediction_error = predicted_rushing_yards - rushing_yards,
    absolute_error = abs(prediction_error)
  )

rb_mae_v3 <- mean(rb_test_results_v3$absolute_error)

rb_rmse_v3 <- sqrt(
  mean(rb_test_results_v3$prediction_error^2)
)

rb_mae_v3
rb_rmse_v3

# Version 4: Injury Features

injury_data_v4 <- load_injuries(2020:2026)

dim(injury_data_v4)

names(injury_data_v4)

head(injury_data_v4, 10)

injury_data_v4 |>
  count(report_status, sort = TRUE)

injury_data_v4 |>
  count(position, sort = TRUE)

injury_data_v4 |>
  count(season, season_type)

injury_data_v4 |>
  count(practice_status, sort = TRUE)

injury_data_v4 |>
  summarise(
    earliest_report = min(date_modified, na.rm = TRUE),
    latest_report = max(date_modified, na.rm = TRUE),
    missing_dates = sum(is.na(date_modified))
  )

injury_data_v4 |>
  count(season, week, team, gsis_id) |>
  filter(n > 1) |>
  arrange(desc(n))

injury_data_v4 |>
  count(season, game_type)

injury_data_v4 |>
  group_by(season) |>
  summarise(
    total_records = n(),
    missing_dates = sum(is.na(date_modified)),
    earliest_date = min(date_modified, na.rm = TRUE),
    latest_date = max(date_modified, na.rm = TRUE),
    .groups = "drop"
  )

injury_data_v4 |>
  filter(
    (season == 2024 & week == 15 & team == "HOU" &
       gsis_id == "00-0039359") |
    (season == 2024 & week == 15 & team == "NYJ" &
       gsis_id == "00-0034270")
  ) |>
  arrange(team, date_modified)

injury_data_v4 |>
  filter(season == 2026) |>
  summarise(
    total_records = n(),
    records_with_dates = sum(!is.na(date_modified)),
    records_without_dates = sum(is.na(date_modified))
  )

injury_data_v4 |>
  filter(season == 2025, game_type == "REG") |>
  select(
    season,
    week,
    team,
    position,
    full_name,
    report_primary_injury,
    report_status,
    practice_status
  ) |>
  head(20)

injury_data_v4 |>
  filter(season == 2025, game_type == "REG") |>
  count(week)

injury_data_v4 |>
  filter(season == 2025, game_type == "REG") |>
  count(position, report_status, sort = TRUE)

injury_data_v4 |>
  filter(
    season == 2025,
    game_type == "REG",
    position == "RB"
  ) |>
  count(practice_status, report_status, sort = TRUE)

injury_data_v4 |>
  filter(season == 2025, game_type == "REG") |>
  count(week, team, gsis_id) |>
  filter(n > 1)

rb_injury_id_check_v4 <- injury_data_v4 |>
  filter(
    game_type == "REG",
    position == "RB"
  ) |>
  inner_join(
    rb_model_ready_v3 |>
      distinct(player_id),
    by = c("gsis_id" = "player_id")
  )

nrow(rb_injury_id_check_v4)

n_distinct(rb_injury_id_check_v4$gsis_id)

rb_injury_id_check_v4 |>
  count(season)

injury_data_v4 |>
  filter(
    game_type == "REG",
    position == "RB"
  ) |>
  count(report_status, practice_status, sort = TRUE)

# Clean injury data

injury_data_v4_clean <- injury_data_v4 |>
  mutate(
    across(
      c(
        report_status,
        practice_status,
        report_primary_injury,
        report_secondary_injury,
        practice_primary_injury,
        practice_secondary_injury
      ),
      ~ na_if(trimws(.x), "")
    )
  )

injury_data_v4_clean |>
  count(report_status, practice_status, sort = TRUE)

rb_injury_duplicates_v4 <- injury_data_v4_clean |>
  filter(
    game_type == "REG",
    position == "RB"
  ) |>
  count(season, week, team, gsis_id) |>
  filter(n > 1) |>
  arrange(desc(n))

rb_injury_duplicates_v4

# Create individual RB injury features

rb_injury_features_v4 <- injury_data_v4_clean |>
  filter(
    game_type == "REG",
    position == "RB"
  ) |>
  mutate(
    rb_report_out = as.integer(report_status == "Out"),
    rb_report_questionable = as.integer(report_status == "Questionable"),
    rb_report_doubtful = as.integer(report_status == "Doubtful"),

    rb_did_not_practice = as.integer(
      practice_status == "Did Not Participate In Practice"
    ),

    rb_limited_practice = as.integer(
      practice_status == "Limited Participation in Practice"
    ),

    rb_full_practice = as.integer(
      practice_status == "Full Participation in Practice"
    )
  ) |>
  select(
    season,
    week,
    team,
    gsis_id,
    full_name,
    report_status,
    practice_status,
    rb_report_out,
    rb_report_questionable,
    rb_report_doubtful,
    rb_did_not_practice,
    rb_limited_practice,
    rb_full_practice
  )

dim(rb_injury_features_v4)

head(rb_injury_features_v4, 10)

rb_injury_features_v4 |>
  summarise(
    total_records = n(),
    total_out = sum(rb_report_out),
    total_questionable = sum(rb_report_questionable),
    total_doubtful = sum(rb_report_doubtful),
    total_did_not_practice = sum(rb_did_not_practice),
    total_limited_practice = sum(rb_limited_practice),
    total_full_practice = sum(rb_full_practice)
  )

# Create individual RB injury features

rb_injury_features_v4 <- injury_data_v4_clean |>
  filter(
    game_type == "REG",
    position == "RB"
  ) |>
  mutate(
    rb_report_out = as.integer(
      coalesce(report_status == "Out", FALSE)
    ),

    rb_report_questionable = as.integer(
      coalesce(report_status == "Questionable", FALSE)
    ),

    rb_report_doubtful = as.integer(
      coalesce(report_status == "Doubtful", FALSE)
    ),

    rb_did_not_practice = as.integer(
      coalesce(
        practice_status == "Did Not Participate In Practice",
        FALSE
      )
    ),

    rb_limited_practice = as.integer(
      coalesce(
        practice_status == "Limited Participation in Practice",
        FALSE
      )
    ),

    rb_full_practice = as.integer(
      coalesce(
        practice_status == "Full Participation in Practice",
        FALSE
      )
    ),

    rb_report_status_missing = as.integer(is.na(report_status)),

    rb_practice_status_missing = as.integer(is.na(practice_status))
  ) |>
  select(
    season,
    week,
    team,
    gsis_id,
    full_name,
    report_status,
    practice_status,
    starts_with("rb_")
  )

rb_injury_features_v4 |>
  summarise(
    total_records = n(),
    total_out = sum(rb_report_out),
    total_questionable = sum(rb_report_questionable),
    total_doubtful = sum(rb_report_doubtful),
    total_did_not_practice = sum(rb_did_not_practice),
    total_limited_practice = sum(rb_limited_practice),
    total_full_practice = sum(rb_full_practice),
    missing_report_status = sum(rb_report_status_missing),
    missing_practice_status = sum(rb_practice_status_missing)
  )

rb_injury_features_v4 |>
  summarise(
    across(
      starts_with("rb_"),
      ~ sum(is.na(.x))
    )
  )

# Join RB injury features to the existing RB model data

rb_model_data_v4 <- rb_model_data_v3 |>
  left_join(
    rb_injury_features_v4 |>
      select(
        season,
        week,
        team,
        gsis_id,
        starts_with("rb_")
      ),
    by = c(
      "season",
      "week",
      "team",
      "player_id" = "gsis_id"
    )
  )

dim(rb_model_data_v3)
dim(rb_model_data_v4)

rb_model_data_v4 |>
  summarise(
    across(
      starts_with("rb_"),
      ~ sum(is.na(.x))
    )
  )

rb_model_data_v4 |>
  summarise(
    total_records = n(),
    records_with_injury_data = sum(!is.na(rb_report_out)),
    records_without_injury_data = sum(is.na(rb_report_out))
  )

rb_model_data_v4 <- rb_model_data_v4 |>
  mutate(
    rb_injury_report_available = as.integer(!is.na(rb_report_out))
  ) |>
  mutate(
    across(
      c(
        rb_report_out,
        rb_report_questionable,
        rb_report_doubtful,
        rb_did_not_practice,
        rb_limited_practice,
        rb_full_practice,
        rb_report_status_missing,
        rb_practice_status_missing
      ),
      ~ coalesce(.x, 0L)
    )
  )

rb_model_data_v4 |>
  summarise(
    total_records = n(),
    records_with_injury_report = sum(rb_injury_report_available),
    records_without_injury_report = sum(rb_injury_report_available == 0),
    missing_injury_features = sum(is.na(rb_report_out))
  )

rb_model_data_v4 |>
  summarise(
    across(
      starts_with("rb_"),
      ~ sum(is.na(.x))
    )
  )

rb_model_data_v4 |>
  count(rb_injury_report_available)

rb_model_data_v4 |>
  filter(rb_injury_report_available == 1) |>
  summarise(
    total = n(),
    out = sum(rb_report_out),
    questionable = sum(rb_report_questionable),
    doubtful = sum(rb_report_doubtful),
    did_not_practice = sum(rb_did_not_practice),
    limited_practice = sum(rb_limited_practice),
    full_practice = sum(rb_full_practice)
  )

rb_injury_features_v4 |>
  count(season, report_status)

rb_injury_features_v4 |>
  count(season, practice_status)

rb_model_data_v4 |>
  filter(rb_injury_report_available == 1) |>
  group_by(season) |>
  summarise(
    total = n(),
    out = sum(rb_report_out),
    questionable = sum(rb_report_questionable),
    doubtful = sum(rb_report_doubtful),
    .groups = "drop"
  )

rb_model_data_v4 |>
  filter(rb_injury_report_available == 1) |>
  group_by(season) |>
  summarise(
    did_not_practice = sum(rb_did_not_practice),
    limited_practice = sum(rb_limited_practice),
    full_practice = sum(rb_full_practice),
    missing_practice = sum(rb_practice_status_missing),
    .groups = "drop"
  )

rb_features_v4 <- c(
  rb_features,
  "rb_report_questionable",
  "rb_did_not_practice",
  "rb_limited_practice",
  "rb_full_practice",
  "rb_injury_report_available",
  "rb_report_status_missing",
  "rb_practice_status_missing"
)

length(rb_features)
length(rb_features_v4)

setdiff(rb_features_v4, names(rb_model_data_v4))

rb_model_data_v4 |>
  filter(season %in% 2020:2025) |>
  group_by(season) |>
  summarise(
    records = n(),
    missing_predictors = sum(
      !complete.cases(pick(all_of(rb_features_v4)))
    ),
    .groups = "drop"
  )

rb_model_data_v4 |>
  filter(season %in% 2020:2025) |>
  group_by(season) |>
  summarise(
    across(
      all_of(rb_features_v4),
      ~ sum(is.na(.x))
    ),
    .groups = "drop"
  )

rb_model_data_v4 |>
  filter(season %in% 2020:2025) |>
  group_by(season) |>
  summarise(
    across(
      all_of(rb_features_v4),
      ~ sum(is.na(.x))
    ),
    .groups = "drop"
  ) |>
  print(width = Inf)
