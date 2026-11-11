# ==========================================
# RB PLAYER PREDICTION MODEL
# ==========================================

# Load required libraries
library(tidyverse)
library(nflreadr)
library(lubridate)
library(slider)
library(xgboost)
library(torch)

# ==========================================
# PROJECT SETTINGS
# ==========================================

first_season <- 2020
current_season <- 2026

seasons <- first_season:current_season

season_types <- c("REG", "POST")

# ==========================================
# LOAD NFL PLAYER STATISTICS
# ==========================================

player_stats <- load_player_stats(
  seasons = seasons,
  summary_level = "week"
)

glimpse(player_stats)

player_stats |>
  count(season_type)

# ==========================================
# FILTER RUNNING BACK STATISTICS
# ==========================================

rb_data <- player_stats |>
  filter(
    position == "RB",
    season_type %in% season_types
  ) |>
  arrange(
    player_id,
    season,
    week
  )

rb_data |> count(season_type)

names(rb_data)

# Create touchdown history features
rb_td_history_features <- rb_data |>
  mutate(
    any_td = as.integer(
      rushing_tds > 0 | receiving_tds > 0
    )
  ) |>
  arrange(player_id, season, week, game_id) |>
  group_by(player_id, season) |>
  mutate(
    # Whether the player scored in their previous appearance
    previous_td = lag(any_td),

    # Consecutive previous appearances with a touchdown
    td_streak = {
      streak <- integer(n())

      for (i in seq_len(n())) {
        if (!is.na(previous_td[i]) && previous_td[i] == 1L) {
          streak[i] <- if (i == 1L) {
            1L
          } else {
            streak[i - 1L] + 1L
          }
        }
      }

      streak
    },

    # Percentage of the previous 3 appearances with a TD
    td_rate_last_3 = slide_dbl(
      previous_td,
      ~ if (anyNA(.x)) NA_real_ else mean(.x),
      .before = 2,
      .complete = TRUE
    ),

    # Percentage of the previous 5 appearances with a TD
    td_rate_last_5 = slide_dbl(
      previous_td,
      ~ if (anyNA(.x)) NA_real_ else mean(.x),
      .before = 4,
      .complete = TRUE
    )
  ) |>
  ungroup() |>
  select(
    player_id,
    game_id,
    td_streak,
    td_rate_last_3,
    td_rate_last_5
  )


rb_data |>
  select(
    player_id,
    player_name,
    headshot_url,
    position,
    team,
    season,
    season_type,
    week,
    carries,
    rushing_yards,
    rushing_tds,
    receptions,
    receiving_yards,
    receiving_tds
  ) |>
  glimpse()

# Check missing values in prediction targets
rb_data |>
  summarise(
    across(
      c(
        carries,
        rushing_yards,
        rushing_tds,
        receptions,
        receiving_yards,
        receiving_tds
      ),
      ~ sum(is.na(.))
    )
  )

# Check duplicate player-team-week records
rb_data |>
  count(
    player_id,
    team,
    season,
    season_type,
    week
  ) |>
  filter(n > 1)

# Load NFL schedules
schedule_data <- load_schedules(
  seasons = seasons
)

# Inspect available schedule columns
names(schedule_data)

schedule_data |>
  count(game_type)

schedule_data |>
  count(game_id) |>
  filter(n > 1)

# Convert schedule into team/opponent format
team_schedule <- clean_homeaway(schedule_data)

# Inspect the resulting columns
names(team_schedule)

# Check the first few records
head(team_schedule)

# Standardize schedule season types
team_schedule <- team_schedule |>
  mutate(
    season_type = if_else(
      game_type == "REG",
      "REG",
      "POST"
    )
  )

# Check for duplicate team-game-week records
team_schedule |>
  count(
    season,
    season_type,
    week,
    team
  ) |>
  filter(n > 1)

# Join RB statistics with team schedule information
rb_data <- rb_data |>
  left_join(
    team_schedule |>
      select(
        game_id,
        season,
        season_type,
        week,
        team,
        gameday,
        opponent,
        location,
        team_rest,
        opponent_rest,
        roof,
        surface,
        temp,
        wind,
        team_qb_id,
        opponent_qb_id
      ),
    by = c(
      "season",
      "season_type",
      "week",
      "team"
    )
  )

names(rb_data)
names(team_schedule)

rb_data |>
  summarise(
    total_records = n(),
    matched_games = sum(!is.na(game_id.y)),
    unmatched_games = sum(is.na(game_id.y)),
    game_id_mismatches = sum(
      game_id.x != game_id.y,
      na.rm = TRUE
    )
  )

rb_data |>
  summarise(
    opponent_matches = sum(
      opponent_team == opponent,
      na.rm = TRUE
    ),
    opponent_mismatches = sum(
      opponent_team != opponent,
      na.rm = TRUE
    ),
    missing_opponents = sum(
      is.na(opponent_team) | is.na(opponent)
    )
  )

# Remove redundant columns and keep one game ID and opponent column
rb_data <- rb_data |>
  select(
    -game_id.y,
    -opponent_team
  ) |>
  rename(
    game_id = game_id.x,
    opponent_team = opponent
  )

# Verify the cleaned columns
names(rb_data)

rb_data |>
  summarise(
    total_records = n(),
    missing_game_ids = sum(is.na(game_id)),
    missing_opponents = sum(is.na(opponent_team)),
    missing_headshots = sum(is.na(headshot_url))
  )

# Convert game date into a Date variable
rb_data <- rb_data |>
  mutate(
    game_date = as.Date(gameday)
  ) |>
  arrange(
    player_id,
    game_date
  )

# Check date quality and chronological coverage
rb_data |>
  summarise(
    total_records = n(),
    missing_dates = sum(is.na(game_date)),
    first_game = min(game_date, na.rm = TRUE),
    last_game = max(game_date, na.rm = TRUE)
  )

rb_data |>
  count(player_id, game_id) |>
  filter(n > 1)

# Load weekly team statistics
team_stats <- load_team_stats(
  seasons = seasons,
  summary_level = "week"
)

# Inspect available columns
names(team_stats)

# Inspect the data structure
glimpse(team_stats)

team_stats |>
  count(game_id, team) |>
  filter(n > 1)

defense_game_stats <- team_stats |>
  transmute(
    season,
    week,
    season_type,
    game_id,
    defense_team = opponent_team,
    opponent_team = team,
    rushing_yards_allowed = rushing_yards,
    carries_allowed = carries,
    rushing_tds_allowed = rushing_tds,
    passing_yards_allowed = passing_yards,
    passing_tds_allowed = passing_tds,
    total_yards_allowed = rushing_yards + passing_yards
  )

glimpse(defense_game_stats)

schedules <- load_schedules(seasons)

glimpse(schedules)

defense_game_stats <- defense_game_stats |>
  left_join(
    schedules |>
      select(game_id, gameday),
    by = "game_id"
  ) |>
  mutate(
    game_date = as.Date(gameday)
  ) |>
  select(-gameday)

defense_game_stats |>
  summarise(
    total_games = n(),
    missing_dates = sum(is.na(game_date)),
    first_game = min(game_date, na.rm = TRUE),
    last_game = max(game_date, na.rm = TRUE)
  )

defense_game_stats <- defense_game_stats |>
  arrange(defense_team, game_date) |>
  group_by(defense_team) |>
  mutate(
    # Previous 3 games
    avg_rushing_yards_allowed_3 = slide_dbl(
      lag(rushing_yards_allowed), mean,
      .before = 2, .complete = TRUE
    ),
    avg_carries_allowed_3 = slide_dbl(
      lag(carries_allowed), mean,
      .before = 2, .complete = TRUE
    ),
    avg_rushing_tds_allowed_3 = slide_dbl(
      lag(rushing_tds_allowed), mean,
      .before = 2, .complete = TRUE
    ),
    avg_passing_yards_allowed_3 = slide_dbl(
      lag(passing_yards_allowed), mean,
      .before = 2, .complete = TRUE
    ),
    avg_total_yards_allowed_3 = slide_dbl(
      lag(total_yards_allowed), mean,
      .before = 2, .complete = TRUE
    ),

    # Previous 5 games
    avg_rushing_yards_allowed_5 = slide_dbl(
      lag(rushing_yards_allowed), mean,
      .before = 4, .complete = TRUE
    ),
    avg_carries_allowed_5 = slide_dbl(
      lag(carries_allowed), mean,
      .before = 4, .complete = TRUE
    ),
    avg_rushing_tds_allowed_5 = slide_dbl(
      lag(rushing_tds_allowed), mean,
      .before = 4, .complete = TRUE
    ),
    avg_passing_yards_allowed_5 = slide_dbl(
      lag(passing_yards_allowed), mean,
      .before = 4, .complete = TRUE
    ),
    avg_total_yards_allowed_5 = slide_dbl(
      lag(total_yards_allowed), mean,
      .before = 4, .complete = TRUE
    )
  ) |>
  ungroup()

glimpse(defense_game_stats)

defense_game_stats <- defense_game_stats |>
  arrange(defense_team, season, game_date) |>
  group_by(defense_team, season) |>
  mutate(
    avg_rushing_yards_allowed = lag(cummean(rushing_yards_allowed)),
    avg_carries_allowed = lag(cummean(carries_allowed)),
    avg_rushing_tds_allowed = lag(cummean(rushing_tds_allowed)),
    avg_passing_yards_allowed = lag(cummean(passing_yards_allowed)),
    avg_passing_tds_allowed = lag(cummean(passing_tds_allowed)),
    avg_total_yards_allowed = lag(cummean(total_yards_allowed))
  ) |>
  ungroup()

glimpse(defense_game_stats)

defense_game_stats <- defense_game_stats |>
  select(-matches("_(3|5)$"))

names(defense_game_stats)

names(rb_data)

rb_data <- rb_data |>
  left_join(
    defense_game_stats |>
      select(
        game_id,
        defense_team,
        avg_rushing_yards_allowed,
        avg_carries_allowed,
        avg_rushing_tds_allowed,
        avg_passing_yards_allowed,
        avg_passing_tds_allowed,
        avg_total_yards_allowed
      ),
    by = c(
      "game_id",
      "opponent_team" = "defense_team"
    )
  )

rb_data |>
  summarise(
    total_records = n(),
    missing_defense_avg = sum(is.na(avg_rushing_yards_allowed)),
    missing_headshots = sum(is.na(headshot_url))
  )

rb_data |>
  filter(is.na(avg_rushing_yards_allowed)) |>
  distinct(season, game_id, opponent_team) |>
  anti_join(
    defense_game_stats |>
      filter(is.na(avg_rushing_yards_allowed)) |>
      transmute(
        season,
        game_id,
        opponent_team = defense_team
      ),
    by = c("season", "game_id", "opponent_team")
  )

names(rb_data)[
  names(rb_data) %in% c(
    "carries",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds"
  )
]

# ==========================================
# RB PREVIOUS GAME PERFORMANCE
# ==========================================

rb_data <- rb_data |>
  arrange(player_id, season, game_date) |>
  group_by(player_id, season) |>
  mutate(
    prev_carries = lag(carries),
    prev_rushing_yards = lag(rushing_yards),
    prev_rushing_tds = lag(rushing_tds),
    prev_receptions = lag(receptions),
    prev_receiving_yards = lag(receiving_yards),
    prev_receiving_tds = lag(receiving_tds)
  ) |>
  ungroup()

rb_data |>
  filter(
    !is.na(prev_carries)
  ) |>
  select(
    player_display_name,
    season,
    week,
    season_type,
    carries,
    prev_carries,
    rushing_yards,
    prev_rushing_yards
  ) |>
  arrange(player_display_name, season, week) |>
  print(n = 10)

rb_data |>
  filter(
    player_display_name == "AJ Dillon",
    season == 2020
  ) |>
  select(week, season_type, carries, prev_carries)

# ==========================================
# RB LAST 3 AND 5 GAME AVERAGES
# ==========================================

rb_data <- rb_data |>
  arrange(player_id, season, game_date) |>
  group_by(player_id, season) |>
  mutate(
    # Carries
    avg_carries_3 = slide_dbl(lag(carries), mean, .before = 2, .complete = TRUE),
    avg_carries_5 = slide_dbl(lag(carries), mean, .before = 4, .complete = TRUE),

    # Rushing yards
    avg_rushing_yards_3 = slide_dbl(lag(rushing_yards), mean, .before = 2, .complete = TRUE),
    avg_rushing_yards_5 = slide_dbl(lag(rushing_yards), mean, .before = 4, .complete = TRUE),

    # Rushing touchdowns
    avg_rushing_tds_3 = slide_dbl(lag(rushing_tds), mean, .before = 2, .complete = TRUE),
    avg_rushing_tds_5 = slide_dbl(lag(rushing_tds), mean, .before = 4, .complete = TRUE),

    # Receptions
    avg_receptions_3 = slide_dbl(lag(receptions), mean, .before = 2, .complete = TRUE),
    avg_receptions_5 = slide_dbl(lag(receptions), mean, .before = 4, .complete = TRUE),

    # Receiving yards
    avg_receiving_yards_3 = slide_dbl(lag(receiving_yards), mean, .before = 2, .complete = TRUE),
    avg_receiving_yards_5 = slide_dbl(lag(receiving_yards), mean, .before = 4, .complete = TRUE),

    # Receiving touchdowns
    avg_receiving_tds_3 = slide_dbl(lag(receiving_tds), mean, .before = 2, .complete = TRUE),
    avg_receiving_tds_5 = slide_dbl(lag(receiving_tds), mean, .before = 4, .complete = TRUE)
  ) |>
  ungroup()

rb_data |>
  filter(player_display_name == "AJ Dillon", season == 2020) |>
  select(
    player_display_name,
    week,
    carries,
    prev_carries,
    avg_carries_3,
    avg_carries_5,
    rushing_yards,
    prev_rushing_yards,
    avg_rushing_yards_3,
    avg_rushing_yards_5
  ) |>
  print(n = 10)

names(rb_data)[
  stringr::str_detect(
    names(rb_data),
    "target|snap|route|carry|touch|opportunity"
  )
]

rb_data |>
  summarise(
    missing_targets = sum(is.na(targets)),
    missing_target_share = sum(is.na(target_share)),
    missing_carries = sum(is.na(carries)),
    missing_receptions = sum(is.na(receptions))
  )

# ==========================================
# RB WORKLOAD AND OPPORTUNITY FEATURES
# ==========================================

rb_data <- rb_data |>
  arrange(player_id, season, game_date) |>
  group_by(player_id, season) |>
  mutate(
    # Total touches
    total_touches = carries + receptions,

    # Previous-game workload
    prev_total_touches = lag(total_touches),
    prev_targets = lag(targets),
    prev_target_share = lag(target_share),

    # Average total touches
    avg_total_touches_3 = slide_dbl(lag(total_touches), mean, .before = 2, .complete = TRUE),
    avg_total_touches_5 = slide_dbl(lag(total_touches), mean, .before = 4, .complete = TRUE),

    # Average targets
    avg_targets_3 = slide_dbl(lag(targets), mean, .before = 2, .complete = TRUE),
    avg_targets_5 = slide_dbl(lag(targets), mean, .before = 4, .complete = TRUE),

    # Average target share
    avg_target_share_3 = slide_dbl(lag(target_share), mean, .before = 2, .complete = TRUE),
    avg_target_share_5 = slide_dbl(lag(target_share), mean, .before = 4, .complete = TRUE)
  ) |>
  ungroup()

rb_data |>
  filter(
    player_display_name == "AJ Dillon",
    season == 2020
  ) |>
  select(
    week,
    carries,
    receptions,
    total_touches,
    prev_total_touches,
    avg_total_touches_3,
    avg_total_touches_5,
    targets,
    prev_targets,
    avg_targets_3,
    avg_targets_5
  ) |>
  print(n = 10)

# ==========================================
# RB SEASON-TO-DATE AVERAGES
# ==========================================

rb_data <- rb_data |>
  arrange(player_id, season, game_date) |>
  group_by(player_id, season) |>
  mutate(
    avg_season_carries = lag(cummean(carries)),
    avg_season_rushing_yards = lag(cummean(rushing_yards)),
    avg_season_rushing_tds = lag(cummean(rushing_tds)),
    avg_season_receptions = lag(cummean(receptions)),
    avg_season_receiving_yards = lag(cummean(receiving_yards)),
    avg_season_receiving_tds = lag(cummean(receiving_tds)),
    avg_season_total_touches = lag(cummean(total_touches)),
    avg_season_targets = lag(cummean(targets)),
    avg_season_target_share = lag(cummean(target_share))
  ) |>
  ungroup()

rb_data |>
  filter(
    player_display_name == "AJ Dillon",
    season == 2020
  ) |>
  select(
    week,
    carries,
    avg_season_carries,
    rushing_yards,
    avg_season_rushing_yards,
    total_touches,
    avg_season_total_touches
  ) |>
  print(n = 10)

names(rb_data)[
  stringr::str_detect(
    names(rb_data),
    "redzone|red_zone|goal|inside_20|inside_10"
  )
]

rb_data |>
  summarise(
    missing = sum(is.na(pt_inside_20)),
    minimum = min(pt_inside_20, na.rm = TRUE),
    maximum = max(pt_inside_20, na.rm = TRUE),
    average = mean(pt_inside_20, na.rm = TRUE)
  )

rb_data |>
  select(
    player_display_name,
    season,
    week,
    carries,
    receptions,
    pt_inside_20
  ) |>
  arrange(desc(pt_inside_20)) |>
  print(n = 10)

names(rb_data)[
  stringr::str_detect(
    names(rb_data),
    "fumble|first_down|rushing_10|rushing_20|rushing_40"
  )
]

names(team_stats)

# ==========================================
# ADDITIONAL DEFENSIVE EFFICIENCY STATS
# ==========================================

defense_game_stats <- defense_game_stats |>
  left_join(
    team_stats |>
      transmute(
        game_id,
        defense_team = opponent_team,
        rushing_epa_allowed = rushing_epa,
        rushing_first_downs_allowed = rushing_first_downs,
        rushing_10_allowed = rushing_10,
        rushing_20_allowed = rushing_20,
        passing_epa_allowed = passing_epa
      ),
    by = c("game_id", "defense_team")
  ) |>
  arrange(defense_team, season, game_date) |>
  group_by(defense_team, season) |>
  mutate(
    avg_rushing_epa_allowed = lag(cummean(rushing_epa_allowed)),
    avg_rushing_first_downs_allowed = lag(cummean(rushing_first_downs_allowed)),
    avg_rushing_10_allowed = lag(cummean(rushing_10_allowed)),
    avg_rushing_20_allowed = lag(cummean(rushing_20_allowed)),
    avg_passing_epa_allowed = lag(cummean(passing_epa_allowed))
  ) |>
  ungroup()

defense_game_stats |>
  summarise(
    total_records = n(),
    missing_rushing_epa = sum(is.na(rushing_epa_allowed)),
    missing_rushing_first_downs = sum(is.na(rushing_first_downs_allowed)),
    missing_rushing_10 = sum(is.na(rushing_10_allowed)),
    missing_rushing_20 = sum(is.na(rushing_20_allowed)),
    missing_passing_epa = sum(is.na(passing_epa_allowed))
  )

# ==========================================
# JOIN ADDITIONAL DEFENSIVE FEATURES TO RB DATA
# ==========================================

rb_data <- rb_data |>
  left_join(
    defense_game_stats |>
      select(
        game_id,
        defense_team,
        avg_rushing_epa_allowed,
        avg_rushing_first_downs_allowed,
        avg_rushing_10_allowed,
        avg_rushing_20_allowed,
        avg_passing_epa_allowed
      ),
    by = c(
      "game_id",
      "opponent_team" = "defense_team"
    )
  )

rb_data |>
  summarise(
    total_records = n(),
    missing_rushing_epa = sum(is.na(avg_rushing_epa_allowed)),
    missing_rushing_first_downs = sum(is.na(avg_rushing_first_downs_allowed)),
    missing_rushing_10 = sum(is.na(avg_rushing_10_allowed)),
    missing_rushing_20 = sum(is.na(avg_rushing_20_allowed)),
    missing_passing_epa = sum(is.na(avg_passing_epa_allowed)),
    missing_headshots = sum(is.na(headshot_url))
  )

rb_data |>
  summarise(
    across(
      c(
        location,
        team_rest,
        opponent_rest,
        roof,
        surface,
        temp,
        wind
      ),
      ~ sum(is.na(.x))
    )
  )

rb_data |>
  group_by(roof) |>
  summarise(
    total_games = n(),
    missing_temp = sum(is.na(temp)),
    missing_wind = sum(is.na(wind)),
    .groups = "drop"
  )

# ==========================================
# TEAM OFFENSIVE PERFORMANCE
# ==========================================

offense_game_stats <- team_stats |>
  transmute(
    season,
    week,
    season_type,
    game_id,
    offense_team = team,
    team_carries = carries,
    team_rushing_yards = rushing_yards,
    team_rushing_tds = rushing_tds,
    team_passing_yards = passing_yards,
    team_passing_tds = passing_tds,
    team_targets = targets,
    team_rushing_epa = rushing_epa,
    team_passing_epa = passing_epa
  ) |>
  left_join(
    schedules |>
      select(game_id, gameday),
    by = "game_id"
  ) |>
  mutate(game_date = as.Date(gameday)) |>
  select(-gameday) |>
  arrange(offense_team, season, game_date) |>
  group_by(offense_team, season) |>
  mutate(
    avg_team_carries = lag(cummean(team_carries)),
    avg_team_rushing_yards = lag(cummean(team_rushing_yards)),
    avg_team_rushing_tds = lag(cummean(team_rushing_tds)),
    avg_team_passing_yards = lag(cummean(team_passing_yards)),
    avg_team_passing_tds = lag(cummean(team_passing_tds)),
    avg_team_targets = lag(cummean(team_targets)),
    avg_team_rushing_epa = lag(cummean(team_rushing_epa)),
    avg_team_passing_epa = lag(cummean(team_passing_epa))
  ) |>
  ungroup()

offense_game_stats |>
  select(
    offense_team,
    season,
    week,
    team_carries,
    avg_team_carries,
    team_rushing_yards,
    avg_team_rushing_yards
  ) |>
  arrange(offense_team, season, week) |>
  print(n = 10)

# ==========================================
# JOIN TEAM OFFENSIVE FEATURES TO RB DATA
# ==========================================

rb_data <- rb_data |>
  left_join(
    offense_game_stats |>
      select(
        game_id,
        offense_team,
        avg_team_carries,
        avg_team_rushing_yards,
        avg_team_rushing_tds,
        avg_team_passing_yards,
        avg_team_passing_tds,
        avg_team_targets,
        avg_team_rushing_epa,
        avg_team_passing_epa
      ),
    by = c(
      "game_id",
      "team" = "offense_team"
    )
  )

rb_data |>
  summarise(
    total_records = n(),
    missing_team_carries = sum(is.na(avg_team_carries)),
    missing_team_rushing_yards = sum(is.na(avg_team_rushing_yards)),
    missing_team_rushing_tds = sum(is.na(avg_team_rushing_tds)),
    missing_team_passing_yards = sum(is.na(avg_team_passing_yards)),
    missing_team_passing_tds = sum(is.na(avg_team_passing_tds)),
    missing_headshots = sum(is.na(headshot_url))
  )

rb_data |>
  summarise(
    missing_headshots = sum(is.na(headshot_url)),
    missing_team_carries = sum(is.na(avg_team_carries))
  )

names(team_stats) |>
  str_subset("sack|pressure|stuff|line|fumble")

team_stats |>
  summarise(
    total_records = n(),
    missing_sacks_suffered = sum(is.na(sacks_suffered)),
    missing_sack_yards_lost = sum(is.na(sack_yards_lost))
  )

rb_data |>
  group_by(season, season_type) |>
  summarise(
    games = n(),
    missing_rushing_yards = sum(is.na(rushing_yards)),
    missing_rushing_tds = sum(is.na(rushing_tds)),
    missing_carries = sum(is.na(carries)),
    missing_receptions = sum(is.na(receptions)),
    missing_receiving_yards = sum(is.na(receiving_yards)),
    missing_receiving_tds = sum(is.na(receiving_tds)),
    .groups = "drop"
  )

sum(is.na(rb_data$receiving_tds))

names(rb_data)


rb_model_data <- rb_data |>
  select(
    # Prediction tracking information
    player_id,
    player_name,
    headshot_url,
    season,
    week,
    season_type,
    game_id,
    game_date,
    team,
    opponent_team,

    # Target
    rushing_yards,

    # Previous-game performance
    prev_carries,
    prev_rushing_yards,
    prev_rushing_tds,
    prev_receptions,
    prev_receiving_yards,
    prev_receiving_tds,

    # Recent performance
    avg_carries_3,
    avg_carries_5,
    avg_rushing_yards_3,
    avg_rushing_yards_5,
    avg_rushing_tds_3,
    avg_rushing_tds_5,
    avg_receptions_3,
    avg_receptions_5,
    avg_receiving_yards_3,
    avg_receiving_yards_5,
    avg_receiving_tds_3,
    avg_receiving_tds_5,

    # Workload and opportunity
    prev_total_touches,
    prev_targets,
    prev_target_share,
    avg_total_touches_3,
    avg_total_touches_5,
    avg_targets_3,
    avg_targets_5,
    avg_target_share_3,
    avg_target_share_5,

    # Season-to-date performance
    avg_season_carries,
    avg_season_rushing_yards,
    avg_season_rushing_tds,
    avg_season_receptions,
    avg_season_receiving_yards,
    avg_season_receiving_tds,
    avg_season_total_touches,
    avg_season_targets,
    avg_season_target_share,

    # Opponent defensive performance
    avg_rushing_yards_allowed,
    avg_carries_allowed,
    avg_rushing_tds_allowed,
    avg_total_yards_allowed,
    avg_rushing_epa_allowed,
    avg_rushing_first_downs_allowed,
    avg_rushing_10_allowed,
    avg_rushing_20_allowed,
    avg_passing_yards_allowed,
    avg_passing_tds_allowed,
    avg_passing_epa_allowed,

    # Team offensive performance
    avg_team_carries,
    avg_team_rushing_yards,
    avg_team_rushing_tds,
    avg_team_passing_yards,
    avg_team_passing_tds,
    avg_team_targets,
    avg_team_rushing_epa,
    avg_team_passing_epa,

    # Game conditions
    location,
    team_rest,
    opponent_rest,
    roof,
    surface,
    temp,
    wind
  ) |>
  mutate(
    across(
      c(season_type, team, opponent_team, location, roof, surface),
      as.factor
    )
  )

dim(rb_model_data)

# Training data: 2020–2023
rb_train <- rb_model_data |>
  filter(season >= 2020, season <= 2023)

# Validation data: 2024
rb_validation <- rb_model_data |>
  filter(season == 2024)

# Testing data: 2025
rb_test <- rb_model_data |>
  filter(season == 2025)

# Current season: 2026
rb_2026 <- rb_model_data |>
  filter(season == 2026)

tibble(
  dataset = c("Training", "Validation", "Testing", "2026 Predictions"),
  records = c(
    nrow(rb_train),
    nrow(rb_validation),
    nrow(rb_test),
    nrow(rb_2026)
  )
)

# Remove information that should not be used as model predictors
rb_feature_data <- rb_model_data |>
  select(
    -player_id,
    -player_name,
    -headshot_url,
    -season,
    -game_id,
    -game_date,
    -team,
    -opponent_team,
    -rushing_yards
  )

# Convert categorical variables into numeric columns
# Keep missing values instead of dropping those records
rb_feature_frame <- model.frame(
  ~ . - 1,
  data = rb_feature_data,
  na.action = na.pass
)

rb_feature_matrix <- model.matrix(
  ~ . - 1,
  data = rb_feature_frame
)

# Separate the target variable
rb_target <- rb_model_data$rushing_yards

# Identify each dataset's rows
train_rows <- rb_model_data$season <= 2023
validation_rows <- rb_model_data$season == 2024
test_rows <- rb_model_data$season == 2025

# Create the feature matrices
X_train <- rb_feature_matrix[train_rows, , drop = FALSE]
X_validation <- rb_feature_matrix[validation_rows, , drop = FALSE]
X_test <- rb_feature_matrix[test_rows, , drop = FALSE]

# Create the target vectors
y_train <- rb_target[train_rows]
y_validation <- rb_target[validation_rows]
y_test <- rb_target[test_rows]

dim(X_train)
dim(X_validation)
dim(X_test)

sum(is.na(X_train))

# Create XGBoost data matrices
dtrain <- xgb.DMatrix(
  data = X_train,
  label = y_train
)

dvalidation <- xgb.DMatrix(
  data = X_validation,
  label = y_validation
)

# Baseline predictions
baseline_prediction <- rep(
  mean(y_train),
  length(y_validation)
)

baseline_mae <- mean(abs(y_validation - baseline_prediction))
baseline_rmse <- sqrt(mean((y_validation - baseline_prediction)^2))

# Set model parameters
rb_params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.03,
  max_depth = 5,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 5,
  tree_method = "hist"
)

# Train the model
set.seed(123)

rb_rush_model <- xgb.train(
  params = rb_params,
  data = dtrain,
  nrounds = 1000,
  evals = list(
    training = dtrain,
    validation = dvalidation
  ),
  early_stopping_rounds = 30,
  verbose = 1
)

# Predict validation rushing yards
xgb_validation_prediction <- predict(
  rb_rush_model,
  dvalidation
)

# Calculate model performance
xgb_validation_mae <- mean(
  abs(y_validation - xgb_validation_prediction)
)

xgb_validation_rmse <- sqrt(
  mean((y_validation - xgb_validation_prediction)^2)
)

# Compare results
tibble(
  model = c("Baseline", "XGBoost"),
  MAE = c(baseline_mae, xgb_validation_mae),
  RMSE = c(baseline_rmse, xgb_validation_rmse)
)

rb_rush_model$best_iteration
tail(rb_rush_model$evaluation_log, 10)

# Check the stored best iteration
attr(rb_rush_model, "best_iteration")

# Check the best validation score
attr(rb_rush_model, "best_score")

# Check the evaluation history
tail(attr(rb_rush_model, "evaluation_log"), 10)

rb_training_history <- attr(rb_rush_model, "evaluation_log")

best_iteration_row <- rb_training_history |>
  slice_min(validation_rmse, n = 1)

best_iteration_row

set.seed(123)

rb_rush_model <- xgb.train(
  params = rb_params,
  data = dtrain,
  nrounds = 150,
  evals = list(
    training = dtrain,
    validation = dvalidation
  ),
  verbose = 0
)

# XGBoost validation predictions
xgb_validation_prediction <- predict(
  rb_rush_model,
  dvalidation
)

# Three-game average baseline
three_game_prediction <- rb_validation$avg_rushing_yards_3

# Compare only records with a complete three-game average
valid_rows <- !is.na(three_game_prediction)

actual <- y_validation[valid_rows]
xgb_pred <- xgb_validation_prediction[valid_rows]
baseline_pred <- three_game_prediction[valid_rows]

# Calculate performance
tibble(
  model = c("Three-game average", "XGBoost"),
  MAE = c(
    mean(abs(actual - baseline_pred)),
    mean(abs(actual - xgb_pred))
  ),
  RMSE = c(
    sqrt(mean((actual - baseline_pred)^2)),
    sqrt(mean((actual - xgb_pred)^2))
  )
)

# Create the 2025 test matrix
dtest <- xgb.DMatrix(
  data = X_test,
  label = y_test
)

# Generate 2025 rushing-yard predictions
xgb_test_prediction <- predict(
  rb_rush_model,
  dtest
)

# Three-game average baseline
baseline_test_prediction <- rb_test$avg_rushing_yards_3

# Compare both models on the same records
valid_test_rows <- !is.na(baseline_test_prediction)

actual_test <- y_test[valid_test_rows]
xgb_test <- xgb_test_prediction[valid_test_rows]
baseline_test <- baseline_test_prediction[valid_test_rows]

# Calculate performance
tibble(
  model = c("Three-game average", "XGBoost"),
  MAE = c(
    mean(abs(actual_test - baseline_test)),
    mean(abs(actual_test - xgb_test))
  ),
  RMSE = c(
    sqrt(mean((actual_test - baseline_test)^2)),
    sqrt(mean((actual_test - xgb_test)^2))
  )
)

rb_feature_importance <- xgb.importance(
  model = rb_rush_model
)

head(rb_feature_importance, 15)

# Change to Neural Network

# Prepare all six RB prediction targets
rb_targets <- rb_data |>
  select(
    player_id,
    game_id,
    carries,
    rushing_yards,
    rushing_tds,
    receptions,
    receiving_yards,
    receiving_tds
  ) |>
  mutate(
    rushing_td_any = as.integer(rushing_tds > 0),
    receiving_td_any = as.integer(receiving_tds > 0)
  ) |>
  select(
    player_id,
    game_id,
    carries,
    rushing_yards,
    rushing_td_any,
    receptions,
    receiving_yards,
    receiving_td_any
  )

# Combine targets with our existing model dataset
rb_multitask_data <- rb_model_data |>
  select(-rushing_yards) |>
  left_join(
    rb_targets,
    by = c("player_id", "game_id")
  )

# Check the results
rb_multitask_data |>
  summarise(
    total_records = n(),
    missing_targets = sum(is.na(carries)) +
      sum(is.na(rushing_yards)) +
      sum(is.na(rushing_td_any)) +
      sum(is.na(receptions)) +
      sum(is.na(receiving_yards)) +
      sum(is.na(receiving_td_any)),
    rushing_td_games = sum(rushing_td_any),
    receiving_td_games = sum(receiving_td_any)
  )

# Check feature and target dimensions
dim(rb_feature_matrix)
dim(rb_multitask_data)

# Verify the first few rows align
head(
  rb_multitask_data |>
    select(player_id, game_id, carries, rushing_yards,
           rushing_td_any, receptions, receiving_yards,
           receiving_td_any)
)

# Check for missing feature values
sum(is.na(rb_feature_matrix))

feature_missing_summary <- tibble(
  feature = colnames(rb_feature_matrix),
  missing_values = colSums(is.na(rb_feature_matrix))
) |>
  filter(missing_values > 0) |>
  arrange(desc(missing_values))

feature_missing_summary

print(feature_missing_summary, n = 60)

# Check missing values in each dataset split
tibble(
  dataset = c("Training", "Validation", "Testing"),
  missing_values = c(
    sum(is.na(rb_feature_matrix[train_rows, , drop = FALSE])),
    sum(is.na(rb_feature_matrix[validation_rows, , drop = FALSE])),
    sum(is.na(rb_feature_matrix[test_rows, , drop = FALSE]))
  )
)

# Check whether any feature is entirely missing in training
colnames(rb_feature_matrix)[
  colSums(!is.na(rb_feature_matrix[train_rows, , drop = FALSE])) == 0
]

# Identify features with missing values
missing_features <- colnames(rb_feature_matrix)[
  colSums(is.na(rb_feature_matrix)) > 0
]

# Calculate medians using training data only
training_medians <- vapply(
  missing_features,
  function(feature) {
    median(
      rb_feature_matrix[train_rows, feature],
      na.rm = TRUE
    )
  },
  numeric(1)
)

# Copy the original feature matrix
rb_feature_matrix_prepared <- rb_feature_matrix

# Create missing-value indicators
missing_indicators <- sapply(
  missing_features,
  function(feature) {
    as.numeric(is.na(rb_feature_matrix[, feature]))
  }
)

# Replace missing values with training medians
for (feature in missing_features) {
  missing_rows <- is.na(rb_feature_matrix_prepared[, feature])

  rb_feature_matrix_prepared[missing_rows, feature] <-
    training_medians[feature]
}

# Add missing-value indicators to the feature matrix
rb_feature_matrix_prepared <- cbind(
  rb_feature_matrix_prepared,
  missing_indicators
)

# Verify the result
dim(rb_feature_matrix_prepared)
sum(is.na(rb_feature_matrix_prepared))

# Separate numerical prediction targets
rb_numeric_targets <- rb_multitask_data |>
  select(
    carries,
    rushing_yards,
    receptions,
    receiving_yards
  )

# Calculate means and standard deviations using training data only
target_means <- sapply(
  rb_numeric_targets[train_rows, ],
  mean
)

target_sds <- sapply(
  rb_numeric_targets[train_rows, ],
  sd
)

# Standardize numerical targets
rb_numeric_targets_scaled <- scale(
  rb_numeric_targets,
  center = target_means,
  scale = target_sds
)

# Prepare binary touchdown targets
rb_td_targets <- rb_multitask_data |>
  select(
    rushing_td_any,
    receiving_td_any
  ) |>
  as.matrix()

# Verify target dimensions
dim(rb_numeric_targets_scaled)
dim(rb_td_targets)

# Check for missing values
sum(is.na(rb_numeric_targets_scaled))
sum(is.na(rb_td_targets))

# Check touchdown outcome distribution
rb_multitask_data |>
  summarise(
    rushing_td_rate = mean(rushing_td_any),
    receiving_td_rate = mean(receiving_td_any),
    rushing_td_positive = sum(rushing_td_any),
    receiving_td_positive = sum(receiving_td_any)
  )

# Convert features to a numeric matrix
rb_x <- as.matrix(rb_feature_matrix_prepared)
storage.mode(rb_x) <- "double"

# Convert numerical targets to a matrix
rb_y_numeric <- as.matrix(rb_numeric_targets_scaled)

# Convert touchdown targets to a matrix
rb_y_td <- as.matrix(rb_td_targets)

# Create training, validation, and testing inputs
X_train_torch <- torch_tensor(rb_x[train_rows, ], dtype = torch_float())
X_validation_torch <- torch_tensor(rb_x[validation_rows, ], dtype = torch_float())
X_test_torch <- torch_tensor(rb_x[test_rows, ], dtype = torch_float())

Y_numeric_train <- torch_tensor(rb_y_numeric[train_rows, ], dtype = torch_float())
Y_numeric_validation <- torch_tensor(rb_y_numeric[validation_rows, ], dtype = torch_float())
Y_numeric_test <- torch_tensor(rb_y_numeric[test_rows, ], dtype = torch_float())

Y_td_train <- torch_tensor(rb_y_td[train_rows, ], dtype = torch_float())
Y_td_validation <- torch_tensor(rb_y_td[validation_rows, ], dtype = torch_float())
Y_td_test <- torch_tensor(rb_y_td[test_rows, ], dtype = torch_float())

# Verify tensor dimensions
X_train_torch$size()
Y_numeric_train$size()
Y_td_train$size()

# Create one combined touchdown target
rb_td_targets <- rb_multitask_data |>
  mutate(
    any_td = as.integer(
      rushing_td_any == 1 | receiving_td_any == 1
    )
  ) |>
  select(any_td) |>
  as.matrix()

# Calculate positive-class weight using training data only
td_weight <- sum(1 - rb_td_targets[train_rows, 1]) /
  sum(rb_td_targets[train_rows, 1])

# Create the touchdown target tensors
Y_td_train <- torch_tensor(
  rb_td_targets[train_rows, , drop = FALSE],
  dtype = torch_float()
)

Y_td_validation <- torch_tensor(
  rb_td_targets[validation_rows, , drop = FALSE],
  dtype = torch_float()
)

Y_td_test <- torch_tensor(
  rb_td_targets[test_rows, , drop = FALSE],
  dtype = torch_float()
)

# Create the neural network
rb_multitask_model <- nn_module(
  "RBMultiTaskModel",
  initialize = function(input_size) {
    self$shared <- nn_sequential(
      nn_linear(input_size, 128),
      nn_relu(),
      nn_dropout(p = 0.2),
      nn_linear(128, 64),
      nn_relu()
    )

    self$numeric_head <- nn_linear(64, 4)
    self$td_head <- nn_linear(64, 1)
  },

  forward = function(x) {
    shared_features <- self$shared(x)

    numeric_predictions <- self$numeric_head(shared_features)
    td_predictions <- self$td_head(shared_features)

    list(
      numeric = numeric_predictions,
      touchdowns = td_predictions
    )
  }
)

# Create the model
rb_model <- rb_multitask_model(
  input_size = ncol(rb_feature_matrix_prepared)
)

# Display the model
rb_model

td_loss_function <- nn_bce_with_logits_loss()

# Numerical prediction loss
numeric_loss_function <- nn_mse_loss()

# Optimizer
optimizer <- optim_adam(
  rb_model$parameters,
  lr = 0.001
)

# Verify the model and class weight
rb_model
td_weight

# Switch model to evaluation mode
rb_model$eval()

# Test with the first 5 training records
with_no_grad({
  test_output <- rb_model(X_train_torch[1:5, ])

  test_numeric_loss <- numeric_loss_function(
    test_output$numeric,
    Y_numeric_train[1:5, ]
  )

  test_td_loss <- td_loss_function(
    test_output$touchdowns,
    Y_td_train[1:5, ]
  )
})

# Check output dimensions
test_output$numeric$size()
test_output$touchdowns$size()

# Display test losses
test_numeric_loss$item()
test_td_loss$item()

# Training settings
epochs <- 100
batch_size <- 128
patience <- 15

# Track training progress
training_history <- tibble(
  epoch = integer(),
  training_loss = double(),
  validation_loss = double()
)

best_validation_loss <- Inf
best_epoch <- 0
epochs_without_improvement <- 0
best_model_state <- NULL

set.seed(123)

# Train the model
for (epoch in 1:epochs) {

  rb_model$train()

  shuffled_rows <- sample(seq_len(nrow(X_train_torch)))

  total_training_loss <- 0

  for (start in seq(1, length(shuffled_rows), by = batch_size)) {

    end <- min(start + batch_size - 1, length(shuffled_rows))
    batch_rows <- shuffled_rows[start:end]

    batch_x <- X_train_torch[batch_rows, ]
    batch_numeric <- Y_numeric_train[batch_rows, ]
    batch_td <- Y_td_train[batch_rows, ]

    optimizer$zero_grad()

    predictions <- rb_model(batch_x)

    numeric_loss <- numeric_loss_function(
      predictions$numeric,
      batch_numeric
    )

    td_loss <- td_loss_function(
      predictions$touchdowns,
      batch_td
    )

    total_loss <- numeric_loss + td_loss

    total_loss$backward()
    optimizer$step()

    total_training_loss <- total_training_loss +
      total_loss$item() * length(batch_rows)
  }

  average_training_loss <-
    total_training_loss / length(shuffled_rows)

  # Evaluate using validation data
  rb_model$eval()

  validation_loss <- with_no_grad({

    validation_predictions <- rb_model(X_validation_torch)

    validation_numeric_loss <- numeric_loss_function(
      validation_predictions$numeric,
      Y_numeric_validation
    )

    validation_td_loss <- td_loss_function(
      validation_predictions$touchdowns,
      Y_td_validation
    )

    (
      validation_numeric_loss +
      validation_td_loss
    )$item()
  })

  # Record results
  training_history <- bind_rows(
    training_history,
    tibble(
      epoch = epoch,
      training_loss = average_training_loss,
      validation_loss = validation_loss
    )
  )

  # Save the best model
  if (validation_loss < best_validation_loss) {

    best_validation_loss <- validation_loss
    best_epoch <- epoch
    epochs_without_improvement <- 0

    best_model_state <- lapply(
      rb_model$state_dict(),
      function(parameter) parameter$clone()
    )

  } else {

    epochs_without_improvement <- epochs_without_improvement + 1
  }

  # Display progress every 5 epochs
  if (epoch %% 5 == 0) {
    cat(
      "Epoch:", epoch,
      "| Training loss:", round(average_training_loss, 4),
      "| Validation loss:", round(validation_loss, 4),
      "\n"
    )
  }

  # Early stopping
  if (epochs_without_improvement >= patience) {
    cat("Early stopping at epoch:", epoch, "\n")
    break
  }
}

# Restore the best-performing model
rb_model$load_state_dict(best_model_state)
rb_model$eval()

# Display final training results
cat("Best epoch:", best_epoch, "\n")
cat("Best validation loss:", best_validation_loss, "\n")

# Generate predictions on the 2025 test data
rb_model$eval()

test_predictions <- with_no_grad({
  rb_model(X_test_torch)
})

# Convert numerical predictions back to original units
predicted_numeric_scaled <- as_array(test_predictions$numeric)

predicted_numeric <- sweep(
  predicted_numeric_scaled,
  2,
  target_sds,
  "*"
)

predicted_numeric <- sweep(
  predicted_numeric,
  2,
  target_means,
  "+"
)

colnames(predicted_numeric) <- c(
  "predicted_carries",
  "predicted_rushing_yards",
  "predicted_receptions",
  "predicted_receiving_yards"
)

# Convert touchdown output into one probability
predicted_td_probabilities <- as_array(
  torch_sigmoid(test_predictions$touchdowns)
)

colnames(predicted_td_probabilities) <- "any_td_probability"

# Display the first five predictions
head(
  cbind(
    predicted_numeric,
    predicted_td_probabilities
  )
)

# Actual numerical outcomes from the 2025 test set
actual_numeric <- as.matrix(
  rb_multitask_data[test_rows, ] |>
    select(carries, rushing_yards, receptions, receiving_yards)
)

# Actual combined touchdown outcomes
actual_td <- rb_td_targets[test_rows, 1]

# Calculate numerical prediction errors
regression_results <- tibble(
  statistic = colnames(actual_numeric),
  MAE = apply(
    abs(predicted_numeric - actual_numeric),
    2,
    mean
  ),
  RMSE = apply(
    (predicted_numeric - actual_numeric)^2,
    2,
    mean
  ) |> sqrt()
)

regression_results

# Calculate combined touchdown classification metrics
td_probability <- as.vector(predicted_td_probabilities)
td_predicted <- as.integer(td_probability >= 0.5)

true_positive <- sum(td_predicted == 1 & actual_td == 1)
false_positive <- sum(td_predicted == 1 & actual_td == 0)
false_negative <- sum(td_predicted == 0 & actual_td == 1)

td_results <- tibble(
  statistic = "Any Touchdown",
  Brier_score = mean((td_probability - actual_td)^2),
  accuracy = mean(td_predicted == actual_td),
  precision = ifelse(
    true_positive + false_positive == 0,
    NA_real_,
    true_positive / (true_positive + false_positive)
  ),
  recall = ifelse(
    true_positive + false_negative == 0,
    NA_real_,
    true_positive / (true_positive + false_negative)
  )
)

td_results

td_baseline_probability <- mean(actual_td)

td_baseline_results <- tibble(
  actual_td_rate = td_baseline_probability,
  model_brier = mean((td_probability - actual_td)^2),
  baseline_brier = mean(
    (td_baseline_probability - actual_td)^2
  ),
  model_accuracy = mean(td_predicted == actual_td),
  majority_class_accuracy = max(
    mean(actual_td == 0),
    mean(actual_td == 1)
  )
)

td_baseline_results

summary(td_probability)

mean(td_probability)

quantile(
  td_probability,
  probs = c(0, 0.25, 0.50, 0.75, 0.90, 0.95, 1)
)

td_calibration <- tibble(
  predicted_probability = td_probability,
  actual_td = actual_td
) |>
  mutate(
    probability_group = ntile(predicted_probability, 10)
  ) |>
  group_by(probability_group) |>
  summarise(
    players = n(),
    average_predicted_probability = mean(predicted_probability),
    actual_td_rate = mean(actual_td),
    .groups = "drop"
  )

td_calibration


# Create the 2025 RB test results spreadsheet
rb_test_results <- rb_multitask_data[test_rows, ] |>
  transmute(
    player = player_name,
    season,
    week,
    season_type,
    game_date,
    team,
    opponent = opponent_team,

    actual_carries = carries,
    actual_rushing_yards = rushing_yards,
    actual_receptions = receptions,
    actual_receiving_yards = receiving_yards
  ) |>
  mutate(
    predicted_carries = predicted_numeric[, 1],
    predicted_rushing_yards = predicted_numeric[, 2],
    predicted_receptions = predicted_numeric[, 3],
    predicted_receiving_yards = predicted_numeric[, 4],

    actual_any_td = as.integer(actual_td),
    predicted_any_td_probability = as.vector(
      predicted_td_probabilities
    ),

    carries_error = abs(predicted_carries - actual_carries),
    rushing_yards_error = abs(
      predicted_rushing_yards - actual_rushing_yards
    ),
    receptions_error = abs(
      predicted_receptions - actual_receptions
    ),
    receiving_yards_error = abs(
      predicted_receiving_yards - actual_receiving_yards
    )
  ) |>
  select(
    player, season, week, season_type, game_date,
    team, opponent,
    actual_carries, predicted_carries, carries_error,
    actual_rushing_yards, predicted_rushing_yards,
    rushing_yards_error,
    actual_receptions, predicted_receptions, receptions_error,
    actual_receiving_yards, predicted_receiving_yards,
    receiving_yards_error,
    actual_any_td, predicted_any_td_probability
  )

# Open the spreadsheet in RStudio
View(rb_test_results)

# Identify all 2026 season records
season_2026_rows <- rb_multitask_data$season == 2026

# Prepare the 2026 input data
X_2026 <- rb_feature_matrix_prepared[
  season_2026_rows,
  ,
  drop = FALSE
]

X_2026_torch <- torch_tensor(
  X_2026,
  dtype = torch_float()
)

# Generate predictions
predictions_2026 <- with_no_grad({
  rb_model(X_2026_torch)
})

# Convert numerical predictions back to original units
predicted_numeric_scaled_2026 <- as_array(
  predictions_2026$numeric
)

predicted_numeric_2026 <- sweep(
  predicted_numeric_scaled_2026,
  2,
  target_sds,
  "*"
)

predicted_numeric_2026 <- sweep(
  predicted_numeric_2026,
  2,
  target_means,
  "+"
)

# Convert touchdown predictions into probabilities
predicted_td_probabilities_2026 <- as.vector(
  as_array(torch_sigmoid(predictions_2026$touchdowns))
)

# Get actual 2026 touchdown outcomes
actual_td_2026 <- as.integer(
  rb_td_targets[season_2026_rows, 1]
)

# Create the 2026 results spreadsheet
rb_2026_results <- rb_multitask_data[season_2026_rows, ] |>
  transmute(
    player = player_name,
    headshot_url,
    season,
    week,
    season_type,
    game_date,
    team,
    opponent = opponent_team,

    actual_carries = carries,
    actual_rushing_yards = rushing_yards,
    actual_receptions = receptions,
    actual_receiving_yards = receiving_yards
  ) |>
  mutate(
    predicted_carries = round(predicted_numeric_2026[, 1], 1),
    predicted_rushing_yards = round(predicted_numeric_2026[, 2], 1),
    predicted_receptions = round(predicted_numeric_2026[, 3], 1),
    predicted_receiving_yards = round(predicted_numeric_2026[, 4], 1),

    actual_any_td = actual_td_2026,

    predicted_any_td_probability = sprintf(
      "%.1f%%",
      predicted_td_probabilities_2026 * 100
    ),

    carries_error = round(
      abs(predicted_carries - actual_carries), 1
    ),

    rushing_yards_error = round(
      abs(predicted_rushing_yards - actual_rushing_yards), 1
    ),

    receptions_error = round(
      abs(predicted_receptions - actual_receptions), 1
    ),

    receiving_yards_error = round(
      abs(predicted_receiving_yards - actual_receiving_yards), 1
    )
  ) |>
  select(
    player, headshot_url, season, week, season_type,
    game_date, team, opponent,
    actual_carries, predicted_carries, carries_error,
    actual_rushing_yards, predicted_rushing_yards,
    rushing_yards_error,
    actual_receptions, predicted_receptions, receptions_error,
    actual_receiving_yards, predicted_receiving_yards,
    receiving_yards_error,
    actual_any_td, predicted_any_td_probability
  )

# Open the spreadsheet
View(rb_2026_results)

# Actual numerical statistics
actual_numeric_2026 <- rb_multitask_data[season_2026_rows, ] |>
  select(
    carries,
    rushing_yards,
    receptions,
    receiving_yards
  ) |>
  as.matrix()

# Calculate MAE and RMSE
rb_2026_regression_metrics <- tibble(
  statistic = colnames(actual_numeric_2026),
  MAE = colMeans(
    abs(predicted_numeric_2026 - actual_numeric_2026)
  ),
  RMSE = sqrt(
    colMeans(
      (predicted_numeric_2026 - actual_numeric_2026)^2
    )
  )
)

# Calculate touchdown performance
predicted_td_class_2026 <- as.integer(
  predicted_td_probabilities_2026 >= 0.5
)

true_positives <- sum(
  predicted_td_class_2026 == 1 & actual_td_2026 == 1
)

predicted_positives <- sum(predicted_td_class_2026 == 1)
actual_positives <- sum(actual_td_2026 == 1)

rb_2026_td_metrics <- tibble(
  metric = c(
    "Brier Score",
    "Accuracy",
    "Precision",
    "Recall",
    "Actual TD Rate",
    "Average Predicted TD Probability"
  ),
  value = c(
    mean((predicted_td_probabilities_2026 - actual_td_2026)^2),
    mean(predicted_td_class_2026 == actual_td_2026),
    ifelse(predicted_positives == 0, NA,
           true_positives / predicted_positives),
    ifelse(actual_positives == 0, NA,
           true_positives / actual_positives),
    mean(actual_td_2026),
    mean(predicted_td_probabilities_2026)
  )
) |>
  mutate(
    value = ifelse(
      metric == "Brier Score",
      sprintf("%.3f", value),
      sprintf("%.1f%%", value * 100)
    )
  )

# Display both evaluation tables
rb_2026_regression_metrics
rb_2026_td_metrics

# Add touchdown history features to the RB model data
rb_model_data_updated <- rb_model_data |>
  left_join(
    rb_td_history_features,
    by = c("player_id", "game_id")
  )

# Check the new columns
rb_model_data_updated |>
  summarise(
    total_records = n(),
    missing_td_streak = sum(is.na(td_streak)),
    missing_td_rate_3 = sum(is.na(td_rate_last_3)),
    missing_td_rate_5 = sum(is.na(td_rate_last_5)),
    longest_td_streak = max(td_streak, na.rm = TRUE)
  )


# Create a table to inspect touchdown history
rb_td_history_check <- rb_model_data_updated |>
  left_join(
    rb_data |>
      transmute(
        player_id,
        game_id,
        actual_any_td = as.integer(
          rushing_tds > 0 | receiving_tds > 0
        )
      ),
    by = c("player_id", "game_id")
  ) |>
  select(
    player_name,
    season,
    week,
    season_type,
    game_date,
    team,
    actual_any_td,
    td_streak,
    td_rate_last_3,
    td_rate_last_5
  ) |>
  arrange(player_name, game_date)

View(rb_td_history_check)

# Add touchdown history to the existing neural network inputs

stopifnot(
  nrow(rb_model_data_updated) == nrow(rb_feature_matrix_prepared)
)

td_history_matrix <- rb_model_data_updated |>
  select(
    td_streak,
    td_rate_last_3,
    td_rate_last_5
  ) |>
  as.matrix()

# Impute missing values using training-set medians
td_history_medians <- apply(
  td_history_matrix[train_rows, , drop = FALSE],
  2,
  median,
  na.rm = TRUE
)

for (feature in colnames(td_history_matrix)) {
  td_history_matrix[
    is.na(td_history_matrix[, feature]),
    feature
  ] <- td_history_medians[feature]
}

# Standardize using training-set statistics
td_history_means <- colMeans(
  td_history_matrix[train_rows, , drop = FALSE]
)

td_history_sds <- apply(
  td_history_matrix[train_rows, , drop = FALSE],
  2,
  sd
)

td_history_sds[td_history_sds == 0 | is.na(td_history_sds)] <- 1

td_history_matrix_scaled <- scale(
  td_history_matrix,
  center = td_history_means,
  scale = td_history_sds
)

# Add the three features to the existing prepared matrix
rb_feature_matrix_prepared_updated <- cbind(
  rb_feature_matrix_prepared,
  td_history_matrix_scaled
)

# Rebuild the neural network input tensors
X_train_torch <- torch_tensor(
  rb_feature_matrix_prepared_updated[train_rows, , drop = FALSE],
  dtype = torch_float()
)

X_validation_torch <- torch_tensor(
  rb_feature_matrix_prepared_updated[validation_rows, , drop = FALSE],
  dtype = torch_float()
)

X_test_torch <- torch_tensor(
  rb_feature_matrix_prepared_updated[test_rows, , drop = FALSE],
  dtype = torch_float()
)

# Confirm the three new features were added
colnames(rb_feature_matrix_prepared_updated)[
  grepl("td_streak|td_rate_last_3|td_rate_last_5",
        colnames(rb_feature_matrix_prepared_updated))
]

# Create updated input tensors
X_train_torch_updated <- torch_tensor(
  rb_feature_matrix_prepared_updated[train_rows, , drop = FALSE],
  dtype = torch_float()
)

X_validation_torch_updated <- torch_tensor(
  rb_feature_matrix_prepared_updated[validation_rows, , drop = FALSE],
  dtype = torch_float()
)

X_test_torch_updated <- torch_tensor(
  rb_feature_matrix_prepared_updated[test_rows, , drop = FALSE],
  dtype = torch_float()
)

# Define updated multitask neural network
rb_multitask_model_updated <- nn_module(
  "rb_multitask_model_updated",

  initialize = function(input_size) {

    self$shared <- nn_sequential(
      nn_linear(input_size, 128),
      nn_relu(),
      nn_dropout(p = 0.2),
      nn_linear(128, 64),
      nn_relu()
    )

    self$numeric_head <- nn_linear(64, 4)
    self$td_head <- nn_linear(64, 1)
  },

  forward = function(x) {

    shared_output <- self$shared(x)

    list(
      numeric = self$numeric_head(shared_output),
      touchdowns = self$td_head(shared_output)
    )
  }
)

# Initialize the updated model
rb_model_updated <- rb_multitask_model_updated(
  input_size = ncol(rb_feature_matrix_prepared_updated)
)

# Define loss functions
numeric_loss_function_updated <- nn_mse_loss()

td_loss_function_updated <- nn_bce_with_logits_loss()

# Define optimizer
optimizer_updated <- optim_adam(
  rb_model_updated$parameters,
  lr = 0.001
)

# Training settings
epochs_updated <- 100
batch_size_updated <- 128
patience_updated <- 15

best_validation_loss_updated <- Inf
epochs_without_improvement_updated <- 0
best_model_state_updated <- NULL

training_history_updated <- tibble(
  epoch = integer(),
  training_loss = double(),
  validation_loss = double()
)

# Training loop
for (epoch in 1:epochs_updated) {

  rb_model_updated$train()

  shuffled_rows <- sample(
    seq_len(nrow(X_train_torch_updated))
  )

  total_training_loss <- 0

  for (start in seq(
    1,
    length(shuffled_rows),
    by = batch_size_updated
  )) {

    batch_rows <- shuffled_rows[
      start:min(
        start + batch_size_updated - 1,
        length(shuffled_rows)
      )
    ]

    x_batch <- X_train_torch_updated[batch_rows, ]
    numeric_batch <- Y_numeric_train[batch_rows, ]
    td_batch <- Y_td_train[batch_rows, ]

    optimizer_updated$zero_grad()

    predictions_batch <- rb_model_updated(x_batch)

    numeric_loss <- numeric_loss_function_updated(
      predictions_batch$numeric,
      numeric_batch
    )

    td_loss <- td_loss_function_updated(
      predictions_batch$touchdowns,
      td_batch
    )

    total_loss <- numeric_loss + td_loss

    total_loss$backward()
    optimizer_updated$step()

    total_training_loss <- total_training_loss +
      as.numeric(total_loss$item())
  }

  average_training_loss <- total_training_loss /
    ceiling(length(shuffled_rows) / batch_size_updated)

  # Calculate validation loss
  rb_model_updated$eval()

  validation_loss_updated <- with_no_grad({

    validation_predictions <- rb_model_updated(
      X_validation_torch_updated
    )

    validation_numeric_loss <- numeric_loss_function_updated(
      validation_predictions$numeric,
      Y_numeric_validation
    )

    validation_td_loss <- td_loss_function_updated(
      validation_predictions$touchdowns,
      Y_td_validation
    )

    validation_numeric_loss + validation_td_loss
  })

  validation_loss_value <- as.numeric(
    validation_loss_updated$item()
  )

  # Save training history
  training_history_updated <- bind_rows(
    training_history_updated,
    tibble(
      epoch = epoch,
      training_loss = average_training_loss,
      validation_loss = validation_loss_value
    )
  )

  # Early stopping
  if (validation_loss_value < best_validation_loss_updated) {

    best_validation_loss_updated <- validation_loss_value

    best_model_state_updated <- lapply(
      rb_model_updated$state_dict(),
      function(parameter) parameter$clone()
    )

    epochs_without_improvement_updated <- 0

  } else {

    epochs_without_improvement_updated <-
      epochs_without_improvement_updated + 1
  }

  cat(
    "Epoch:", epoch,
    "| Training Loss:", round(average_training_loss, 4),
    "| Validation Loss:", round(validation_loss_value, 4),
    "\n"
  )

  if (epochs_without_improvement_updated >= patience_updated) {
    cat("Early stopping triggered at epoch", epoch, "\n")
    break
  }
}

# Restore the best model weights
rb_model_updated$load_state_dict(
  best_model_state_updated
)

# Display training history
training_history_updated

# Best validation loss
best_validation_loss_updated

# ============================================================
# UPDATED 2025 TEST PREDICTIONS
# ============================================================

rb_model_updated$eval()

test_predictions_updated <- with_no_grad({
  rb_model_updated(X_test_torch_updated)
})

# Convert numeric predictions back to original units
predicted_numeric_scaled_updated <- as_array(
  test_predictions_updated$numeric
)

predicted_numeric_updated <- sweep(
  predicted_numeric_scaled_updated,
  2,
  target_sds,
  "*"
)

predicted_numeric_updated <- sweep(
  predicted_numeric_updated,
  2,
  target_means,
  "+"
)

# Convert touchdown logits into probabilities
predicted_td_probabilities_updated <- as.vector(
  as_array(
    torch_sigmoid(test_predictions_updated$touchdowns)
  )
)

# Create updated 2025 results spreadsheet
rb_test_results_updated <- rb_multitask_data[test_rows, ] |>
  transmute(
    player_id,
    game_id,
    player = player_name,
    season,
    week,
    season_type,
    game_date,
    team,
    opponent = opponent_team,

    actual_carries = carries,
    actual_rushing_yards = rushing_yards,
    actual_receptions = receptions,
    actual_receiving_yards = receiving_yards
  ) |>
  mutate(
    predicted_carries = predicted_numeric_updated[, 1],
    predicted_rushing_yards = predicted_numeric_updated[, 2],
    predicted_receptions = predicted_numeric_updated[, 3],
    predicted_receiving_yards = predicted_numeric_updated[, 4],

    actual_any_td = as.integer(
      rb_td_targets[test_rows, 1]
    ),

    predicted_any_td_probability =
      predicted_td_probabilities_updated,

    carries_error = abs(
      predicted_carries - actual_carries
    ),

    rushing_yards_error = abs(
      predicted_rushing_yards - actual_rushing_yards
    ),

    receptions_error = abs(
      predicted_receptions - actual_receptions
    ),

    receiving_yards_error = abs(
      predicted_receiving_yards - actual_receiving_yards
    )
  ) |>
  left_join(
    rb_model_data_updated |>
      select(
        player_id,
        game_id,
        td_streak,
        td_rate_last_3,
        td_rate_last_5
      ),
    by = c("player_id", "game_id")
  ) |>
  select(
    -player_id,
    -game_id,
    player,
    season,
    week,
    season_type,
    game_date,
    team,
    opponent,
    td_streak,
    td_rate_last_3,
    td_rate_last_5,
    actual_carries,
    predicted_carries,
    carries_error,
    actual_rushing_yards,
    predicted_rushing_yards,
    rushing_yards_error,
    actual_receptions,
    predicted_receptions,
    receptions_error,
    actual_receiving_yards,
    predicted_receiving_yards,
    receiving_yards_error,
    actual_any_td,
    predicted_any_td_probability
  )

View(rb_test_results_updated)


# ============================================================
# UPDATED 2026 PREDICTIONS
# ============================================================

# Identify 2026 records
season_2026_rows <- rb_multitask_data$season == 2026

# Prepare updated 2026 inputs
X_2026_updated <- rb_feature_matrix_prepared_updated[
  season_2026_rows,
  ,
  drop = FALSE
]

X_2026_torch_updated <- torch_tensor(
  X_2026_updated,
  dtype = torch_float()
)

# Generate predictions
predictions_2026_updated <- with_no_grad({
  rb_model_updated(X_2026_torch_updated)
})

# Convert numeric predictions back to original units
predicted_numeric_scaled_2026_updated <- as_array(
  predictions_2026_updated$numeric
)

predicted_numeric_2026_updated <- sweep(
  predicted_numeric_scaled_2026_updated,
  2,
  target_sds,
  "*"
)

predicted_numeric_2026_updated <- sweep(
  predicted_numeric_2026_updated,
  2,
  target_means,
  "+"
)

# Convert touchdown logits into probabilities
predicted_td_probabilities_2026_updated <- as.vector(
  as_array(
    torch_sigmoid(predictions_2026_updated$touchdowns)
  )
)

# Actual 2026 touchdown outcomes
actual_td_2026_updated <- as.integer(
  rb_td_targets[season_2026_rows, 1]
)

# Create updated 2026 spreadsheet
rb_2026_results_updated <- rb_multitask_data[season_2026_rows, ] |>
  transmute(
    player_id,
    game_id,
    player = player_name,
    headshot_url,
    season,
    week,
    season_type,
    game_date,
    team,
    opponent = opponent_team,

    actual_carries = carries,
    actual_rushing_yards = rushing_yards,
    actual_receptions = receptions,
    actual_receiving_yards = receiving_yards
  ) |>
  mutate(
    predicted_carries = round(
      predicted_numeric_2026_updated[, 1], 1
    ),

    predicted_rushing_yards = round(
      predicted_numeric_2026_updated[, 2], 1
    ),

    predicted_receptions = round(
      predicted_numeric_2026_updated[, 3], 1
    ),

    predicted_receiving_yards = round(
      predicted_numeric_2026_updated[, 4], 1
    ),

    actual_any_td = actual_td_2026_updated,

    predicted_any_td_probability = sprintf(
      "%.1f%%",
      predicted_td_probabilities_2026_updated * 100
    ),

    carries_error = round(
      abs(predicted_carries - actual_carries), 1
    ),

    rushing_yards_error = round(
      abs(predicted_rushing_yards - actual_rushing_yards), 1
    ),

    receptions_error = round(
      abs(predicted_receptions - actual_receptions), 1
    ),

    receiving_yards_error = round(
      abs(predicted_receiving_yards - actual_receiving_yards), 1
    )
  ) |>
  left_join(
    rb_model_data_updated |>
      select(
        player_id,
        game_id,
        td_streak,
        td_rate_last_3,
        td_rate_last_5
      ),
    by = c("player_id", "game_id")
  ) |>
  select(
    -player_id,
    -game_id,
    player,
    headshot_url,
    season,
    week,
    season_type,
    game_date,
    team,
    opponent,
    td_streak,
    td_rate_last_3,
    td_rate_last_5,
    actual_carries,
    predicted_carries,
    carries_error,
    actual_rushing_yards,
    predicted_rushing_yards,
    rushing_yards_error,
    actual_receptions,
    predicted_receptions,
    receptions_error,
    actual_receiving_yards,
    predicted_receiving_yards,
    receiving_yards_error,
    actual_any_td,
    predicted_any_td_probability
  )

View(rb_2026_results_updated)

# Confirm the new model has the expected input count
ncol(rb_feature_matrix_prepared_updated)

# Confirm all three TD-history features are included
colnames(rb_feature_matrix_prepared_updated)[
  grepl(
    "td_streak|td_rate_last_3|td_rate_last_5",
    colnames(rb_feature_matrix_prepared_updated)
  )
]

# Check the updated prediction spreadsheets
head(
  rb_2026_results_updated |>
    select(
      player,
      td_streak,
      td_rate_last_3,
      td_rate_last_5,
      predicted_any_td_probability
    )
)

summary(predicted_td_probabilities_2026_updated)

mean(predicted_td_probabilities_2026_updated)

quantile(
  predicted_td_probabilities_2026_updated,
  probs = c(0, 0.25, 0.50, 0.75, 0.90, 0.95, 1)
)

rb_2026_results_updated |>
  group_by(td_streak) |>
  summarise(
    players = n(),
    average_td_probability = mean(
      predicted_td_probabilities_2026_updated,
      na.rm = TRUE
    ),
    .groups = "drop"
  )

rb_2026_results_updated |>
  mutate(
    td_probability_numeric =
      predicted_td_probabilities_2026_updated
  ) |>
  group_by(td_streak) |>
  summarise(
    players = n(),
    average_td_probability = mean(
      td_probability_numeric,
      na.rm = TRUE
    ),
    .groups = "drop"
  ) |>
  mutate(
    average_td_probability = round(
      average_td_probability * 100,
      2
    )
  )

rb_2026_results_updated |>
  mutate(
    td_probability_numeric =
      predicted_td_probabilities_2026_updated
  ) |>
  group_by(td_rate_last_3) |>
  summarise(
    players = n(),
    average_td_probability = mean(
      td_probability_numeric,
      na.rm = TRUE
    ),
    .groups = "drop"
  ) |>
  arrange(td_rate_last_3)

# Compare MAE and RMSE: Original vs Updated Model (2025)

# Actual numerical stats from the 2025 test set
numeric_stat_names <- c(
  "carries",
  "rushing_yards",
  "receptions",
  "receiving_yards"
)

actual_numeric_2025 <- rb_multitask_data[test_rows, ] |>
  select(all_of(numeric_stat_names)) |>
  as.matrix()

# Predictions from both models
original_predictions_2025 <- as.matrix(predicted_numeric)
updated_predictions_2025 <- as.matrix(predicted_numeric_updated)

# Make sure columns use the correct stat names
colnames(original_predictions_2025) <- numeric_stat_names
colnames(updated_predictions_2025) <- numeric_stat_names

# Function to calculate MAE and RMSE
calculate_regression_metrics <- function(actual, predicted, model_name) {

  tibble(
    stat = numeric_stat_names,
    MAE = colMeans(abs(actual - predicted)),
    RMSE = sqrt(colMeans((actual - predicted)^2)),
    model = model_name
  )
}

# Calculate metrics for both models
mae_rmse_comparison <- bind_rows(
  calculate_regression_metrics(
    actual_numeric_2025,
    original_predictions_2025,
    "Original Model"
  ),
  calculate_regression_metrics(
    actual_numeric_2025,
    updated_predictions_2025,
    "Updated Model"
  )
)

# Display results
mae_rmse_comparison |>
  mutate(
    MAE = round(MAE, 2),
    RMSE = round(RMSE, 2)
  )

# Compare TD Prediction Performance: Original vs Updated Model

# Actual touchdown outcomes for the 2025 test set
actual_td_2025 <- as.numeric(rb_td_targets[test_rows, 1])

# Function to calculate touchdown metrics
calculate_td_metrics <- function(actual, predicted, model_name) {

  predicted_class <- ifelse(predicted >= 0.5, 1, 0)

  true_positive <- sum(predicted_class == 1 & actual == 1)
  false_positive <- sum(predicted_class == 1 & actual == 0)
  false_negative <- sum(predicted_class == 0 & actual == 1)

  tibble(
    model = model_name,
    Brier_Score = mean((predicted - actual)^2),
    Accuracy = mean(predicted_class == actual),
    Precision = ifelse(
      true_positive + false_positive == 0,
      0,
      true_positive / (true_positive + false_positive)
    ),
    Recall = ifelse(
      true_positive + false_negative == 0,
      0,
      true_positive / (true_positive + false_negative)
    )
  )
}

# Compare both models
td_performance_comparison <- bind_rows(
  calculate_td_metrics(
    actual_td_2025,
    predicted_td_probabilities,
    "Original Model"
  ),
  calculate_td_metrics(
    actual_td_2025,
    predicted_td_probabilities_updated,
    "Updated Model"
  )
)

# Display results as percentages, except Brier score
td_performance_comparison |>
  mutate(
    Brier_Score = round(Brier_Score, 4),
    Accuracy = round(Accuracy * 100, 2),
    Precision = round(Precision * 100, 2),
    Recall = round(Recall * 100, 2)
  )
