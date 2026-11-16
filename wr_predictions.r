# ==========================================
# WR PREDICTIONS MODEL
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
# LOAD WR DATA
# ==========================================

wr_data <- load_player_stats(
  seasons = 2020:2026,
  summary_level = "week"
)

# Filter for wide receivers
wr_data <- wr_data %>%
  filter(position == "WR")

dim(wr_data)
names(wr_data)
head(wr_data)

# Check season types
wr_data %>%
  count(season_type)

# Check records by season
wr_data %>%
  count(season, season_type)

# Check receiving statistics
wr_data %>%
  summarise(
    total_records = n(),
    records_with_targets = sum(targets > 0, na.rm = TRUE),
    records_with_receptions = sum(receptions > 0, na.rm = TRUE),
    records_with_receiving_yards = sum(receiving_yards > 0, na.rm = TRUE),
    records_with_touchdowns = sum(receiving_tds > 0, na.rm = TRUE)
  )

# ==========================================
# CLEAN WR DATA
# ==========================================

wr_model_data <- wr_data %>%
  filter(season_type == "REG") %>%
  select(
    player_id,
    player_name,
    player_display_name,
    position,
    season,
    week,
    game_id,
    team,
    opponent_team,
    receiving_yards,
    receiving_tds,
    targets,
    receptions,
    rushing_yards,
    carries
  ) %>%
  arrange(player_id, season, week)

dim(wr_model_data)

names(wr_model_data)

head(wr_model_data)

# ==========================================
# CHECK DUPLICATE WR RECORDS
# ==========================================

wr_model_data %>%
  count(player_id, season, week) %>%
  filter(n > 1)

n_distinct(wr_model_data$player_id)

# ==========================================
# WR PLAYER HISTORY FEATURES
# ==========================================

wr_model_data <- wr_model_data %>%
  group_by(player_id) %>%
  arrange(season, week, .by_group = TRUE) %>%
  mutate(
    # Previous game statistics
    prev_receiving_yards = lag(receiving_yards),
    prev_receiving_tds = lag(receiving_tds),
    prev_targets = lag(targets),
    prev_receptions = lag(receptions),
    prev_rushing_yards = lag(rushing_yards),
    prev_carries = lag(carries),

    # Average statistics over previous 3 games
    avg_receiving_yards_3 = lag(slider::slide_dbl(
      receiving_yards, mean, .before = 2, .complete = FALSE
    )),
    avg_receiving_tds_3 = lag(slider::slide_dbl(
      receiving_tds, mean, .before = 2, .complete = FALSE
    )),
    avg_targets_3 = lag(slider::slide_dbl(
      targets, mean, .before = 2, .complete = FALSE
    )),
    avg_receptions_3 = lag(slider::slide_dbl(
      receptions, mean, .before = 2, .complete = FALSE
    )),
    avg_rushing_yards_3 = lag(slider::slide_dbl(
      rushing_yards, mean, .before = 2, .complete = FALSE
    )),
    avg_carries_3 = lag(slider::slide_dbl(
      carries, mean, .before = 2, .complete = FALSE
    )),

    # Average statistics over previous 5 games
    avg_receiving_yards_5 = lag(slider::slide_dbl(
      receiving_yards, mean, .before = 4, .complete = FALSE
    )),
    avg_receiving_tds_5 = lag(slider::slide_dbl(
      receiving_tds, mean, .before = 4, .complete = FALSE
    )),
    avg_targets_5 = lag(slider::slide_dbl(
      targets, mean, .before = 4, .complete = FALSE
    )),
    avg_receptions_5 = lag(slider::slide_dbl(
      receptions, mean, .before = 4, .complete = FALSE
    )),
    avg_rushing_yards_5 = lag(slider::slide_dbl(
      rushing_yards, mean, .before = 4, .complete = FALSE
    )),
    avg_carries_5 = lag(slider::slide_dbl(
      carries, mean, .before = 4, .complete = FALSE
    ))
  ) %>%
  ungroup()

wr_model_data %>%
  select(
    player_display_name,
    season,
    week,
    receiving_yards,
    prev_receiving_yards,
    avg_receiving_yards_3,
    avg_receiving_yards_5
  ) %>%
  filter(player_display_name == "Justin Jefferson") %>%
  head(10)

# ==========================================
# WR TOUCHDOWN HISTORY FEATURES
# ==========================================

# Calculate consecutive previous games with a receiving TD
get_td_streak <- function(td) {
  streak <- integer(length(td))
  current_streak <- 0L

  for (i in seq_along(td)) {
    streak[i] <- current_streak

    if (is.na(td[i]) || td[i] == 0) {
      current_streak <- 0L
    } else {
      current_streak <- current_streak + 1L
    }
  }

  streak
}

wr_model_data <- wr_model_data %>%
  group_by(player_id) %>%
  arrange(season, week, .by_group = TRUE) %>%
  mutate(
    # Previous game TD information
    prev_receiving_tds = lag(receiving_tds),
    prev_td_indicator = lag(as.integer(receiving_tds > 0)),

    # Consecutive previous games with a receiving TD
    td_streak = get_td_streak(
      as.integer(receiving_tds > 0)
    ),

    # TD rate over previous 3 games
    td_rate_last_3 = lag(
      slider::slide_dbl(
        as.integer(receiving_tds > 0),
        mean,
        .before = 2,
        .complete = FALSE
      )
    ),

    # TD rate over previous 5 games
    td_rate_last_5 = lag(
      slider::slide_dbl(
        as.integer(receiving_tds > 0),
        mean,
        .before = 4,
        .complete = FALSE
      )
    )
  ) %>%
  ungroup()

wr_model_data %>%
  select(
    player_display_name,
    season,
    week,
    receiving_tds,
    prev_receiving_tds,
    prev_td_indicator,
    td_streak,
    td_rate_last_3,
    td_rate_last_5
  ) %>%
  filter(player_display_name == "Justin Jefferson") %>%
  head(15)

# ==========================================
# CHECK WR HISTORY MISSING VALUES
# ==========================================

wr_model_data %>%
  summarise(
    across(
      c(
        starts_with("prev_"),
        starts_with("avg_"),
        td_streak,
        td_rate_last_3,
        td_rate_last_5
      ),
      ~sum(is.na(.))
    )
  )

# ==========================================
# LOAD WEEKLY TEAM STATISTICS
# ==========================================

wr_team_stats <- load_team_stats(
  seasons = 2020:2026,
  summary_level = "week"
)

# Keep regular-season games
wr_team_stats <- wr_team_stats %>%
  filter(season_type == "REG")

dim(wr_team_stats)

names(wr_team_stats)

head(wr_team_stats)

# ==========================================
# WR OPPONENT DEFENSE DATA
# ==========================================

wr_defense_data <- wr_team_stats %>%
  transmute(
    season,
    week,
    game_id,
    defense_team = opponent_team,

    # Passing statistics allowed
    pass_yards_allowed = passing_yards,
    pass_tds_allowed = passing_tds,
    pass_attempts_allowed = attempts,
    pass_completions_allowed = completions,
    pass_epa_allowed = passing_epa,

    # Additional passing defense statistics
    interceptions_made = passing_interceptions,
    sacks_recorded = sacks_suffered,

    # Rushing statistics allowed
    rush_yards_allowed = rushing_yards,
    rush_tds_allowed = rushing_tds,
    rush_attempts_allowed = carries,
    rush_epa_allowed = rushing_epa
  ) %>%
  arrange(defense_team, season, week)

dim(wr_defense_data)
head(wr_defense_data)

wr_defense_data %>%
  count(defense_team, season, week) %>%
  filter(n > 1)

# ==========================================
# LOAD DEFENSIVE PLAYER STATISTICS
# ==========================================

wr_defensive_players <- load_player_stats(
  seasons = 2020:2026,
  summary_level = "week"
) %>%
  filter(
    season_type == "REG",
    position == "CB"
  )

dim(wr_defensive_players)
names(wr_defensive_players)
head(wr_defensive_players)

# ==========================================
# LOAD NEXT GEN STATS
# ==========================================

wr_ngs_receiving <- load_nextgen_stats(
  seasons = 2020:2026,
  stat_type = "receiving"
)

dim(wr_ngs_receiving)
names(wr_ngs_receiving)
head(wr_ngs_receiving)

# ==========================================
# PREPARE WR NEXT GEN STATS FEATURES
# ==========================================

wr_ngs_features <- wr_ngs_receiving %>%
  filter(
    season_type == "REG",
    player_position == "WR",
    week > 0
  ) %>%
  select(
    player_id = player_gsis_id,
    season,
    week,
    avg_separation,
    avg_cushion,
    avg_intended_air_yards,
    percent_share_of_intended_air_yards,
    avg_yac,
    avg_yac_above_expectation
  ) %>%
  arrange(player_id, season, week)

wr_ngs_features %>%
  count(player_id, season, week) %>%
  filter(n > 1)

# ==========================================
# JOIN NEXT GEN STATS TO WR DATA
# ==========================================

wr_model_data <- wr_model_data %>%
  left_join(
    wr_ngs_features,
    by = c("player_id", "season", "week")
  )

dim(wr_model_data)

wr_model_data %>%
  summarise(
    across(
      c(
        avg_separation,
        avg_cushion,
        avg_intended_air_yards,
        percent_share_of_intended_air_yards,
        avg_yac,
        avg_yac_above_expectation
      ),
      ~sum(!is.na(.))
    )
  )

# ==========================================
# WR NEXT GEN STATS HISTORY FEATURES
# ==========================================

wr_model_data <- wr_model_data %>%
  group_by(player_id) %>%
  arrange(season, week, .by_group = TRUE) %>%
  mutate(
    # Previous game NGS statistics
    ngs_prev_separation = lag(avg_separation),
    ngs_prev_cushion = lag(avg_cushion),
    ngs_prev_air_yards = lag(avg_intended_air_yards),
    ngs_prev_air_yards_share = lag(percent_share_of_intended_air_yards),
    ngs_prev_yac = lag(avg_yac),
    ngs_prev_yac_above_expectation = lag(avg_yac_above_expectation),

    # Previous 3-game NGS averages
    ngs_avg_separation_3 = lag(slider::slide_dbl(
      avg_separation, ~if (all(is.na(.x))) NA_real_ else mean(.x, na.rm = TRUE),
      .before = 2, .complete = FALSE
    )),
    ngs_avg_cushion_3 = lag(slider::slide_dbl(
      avg_cushion, ~if (all(is.na(.x))) NA_real_ else mean(.x, na.rm = TRUE),
      .before = 2, .complete = FALSE
    )),
    ngs_avg_air_yards_3 = lag(slider::slide_dbl(
      avg_intended_air_yards, ~if (all(is.na(.x))) NA_real_ else mean(.x, na.rm = TRUE),
      .before = 2, .complete = FALSE
    )),
    ngs_avg_air_yards_share_3 = lag(slider::slide_dbl(
      percent_share_of_intended_air_yards, ~if (all(is.na(.x))) NA_real_ else mean(.x, na.rm = TRUE),
      .before = 2, .complete = FALSE
    )),
    ngs_avg_yac_3 = lag(slider::slide_dbl(
      avg_yac, ~if (all(is.na(.x))) NA_real_ else mean(.x, na.rm = TRUE),
      .before = 2, .complete = FALSE
    )),
    ngs_avg_yac_above_expectation_3 = lag(slider::slide_dbl(
      avg_yac_above_expectation, ~if (all(is.na(.x))) NA_real_ else mean(.x, na.rm = TRUE),
      .before = 2, .complete = FALSE
    )),

    # Previous 5-game NGS averages
    ngs_avg_separation_5 = lag(slider::slide_dbl(
      avg_separation, ~if (all(is.na(.x))) NA_real_ else mean(.x, na.rm = TRUE),
      .before = 4, .complete = FALSE
    )),
    ngs_avg_cushion_5 = lag(slider::slide_dbl(
      avg_cushion, ~if (all(is.na(.x))) NA_real_ else mean(.x, na.rm = TRUE),
      .before = 4, .complete = FALSE
    )),
    ngs_avg_air_yards_5 = lag(slider::slide_dbl(
      avg_intended_air_yards, ~if (all(is.na(.x))) NA_real_ else mean(.x, na.rm = TRUE),
      .before = 4, .complete = FALSE
    )),
    ngs_avg_air_yards_share_5 = lag(slider::slide_dbl(
      percent_share_of_intended_air_yards, ~if (all(is.na(.x))) NA_real_ else mean(.x, na.rm = TRUE),
      .before = 4, .complete = FALSE
    )),
    ngs_avg_yac_5 = lag(slider::slide_dbl(
      avg_yac, ~if (all(is.na(.x))) NA_real_ else mean(.x, na.rm = TRUE),
      .before = 4, .complete = FALSE
    )),
    ngs_avg_yac_above_expectation_5 = lag(slider::slide_dbl(
      avg_yac_above_expectation, ~if (all(is.na(.x))) NA_real_ else mean(.x, na.rm = TRUE),
      .before = 4, .complete = FALSE
    ))
  ) %>%
  ungroup()

# Restore weekly NGS statistics for history calculations
wr_model_data <- wr_model_data %>%
  left_join(
    wr_ngs_features,
    by = c("player_id", "season", "week")
  )

wr_model_data %>%
  filter(player_display_name == "Justin Jefferson") %>%
  select(
    season,
    week,
    ngs_prev_separation,
    ngs_avg_separation_3,
    ngs_avg_separation_5,
    ngs_prev_yac,
    ngs_avg_yac_3,
    ngs_avg_yac_5
  ) %>%
  head(10)

wr_model_data <- wr_model_data %>%
  select(
    -any_of(c(
      "avg_separation",
      "avg_cushion",
      "avg_intended_air_yards",
      "percent_share_of_intended_air_yards",
      "avg_yac",
      "avg_yac_above_expectation"
    ))
  )

dim(wr_model_data)
names(wr_model_data)

# ==========================================
# WR OPPONENT DEFENSE HISTORY FEATURES
# ==========================================

wr_defense_history <- wr_defense_data %>%
  group_by(defense_team) %>%
  arrange(season, week, .by_group = TRUE) %>%
  mutate(
    across(
      c(
        pass_yards_allowed,
        pass_tds_allowed,
        pass_attempts_allowed,
        pass_completions_allowed,
        pass_epa_allowed,
        interceptions_made,
        sacks_recorded,
        rush_yards_allowed,
        rush_tds_allowed,
        rush_attempts_allowed,
        rush_epa_allowed
      ),
      list(
        prev = ~lag(.x),
        avg3 = ~lag(slider::slide_dbl(
          .x, mean, .before = 2, .complete = FALSE
        )),
        avg5 = ~lag(slider::slide_dbl(
          .x, mean, .before = 4, .complete = FALSE
        ))
      ),
      .names = "{.fn}_{.col}"
    )
  ) %>%
  ungroup()

# ==========================================
# CLEAN WR DEFENSE HISTORY DATA
# ==========================================

wr_defense_history <- wr_defense_history %>%
  select(
    defense_team,
    season,
    week,
    starts_with("prev_"),
    starts_with("avg3_"),
    starts_with("avg5_")
  )

# ==========================================
# JOIN OPPONENT DEFENSE HISTORY
# ==========================================

wr_model_data <- wr_model_data %>%
  left_join(
    wr_defense_history,
    by = c(
      "opponent_team" = "defense_team",
      "season",
      "week"
    )
  )

dim(wr_model_data)

wr_model_data %>%
  summarise(
    across(
      starts_with("prev_"),
      ~sum(!is.na(.))
    )
  )

# ==========================================
# LOAD CORNERBACK STATISTICS
# ==========================================

wr_cb_data <- load_player_stats(
  seasons = 2020:2026,
  summary_level = "week"
) %>%
  filter(
    season_type == "REG",
    position == "CB"
  )

dim(wr_cb_data)

names(wr_cb_data)

head(wr_cb_data)

wr_cb_data %>%
  group_by(player_display_name, team) %>%
  summarise(
    interceptions = sum(def_interceptions, na.rm = TRUE),
    passes_defended = sum(def_pass_defended, na.rm = TRUE),
    tackles = sum(def_tackles_solo, na.rm = TRUE),
    games_played = n_distinct(game_id),
    .groups = "drop"
  ) %>%
  arrange(desc(passes_defended)) %>%
  head(15)

# ==========================================
# CORNERBACK PERFORMANCE BY SEASON
# ==========================================

wr_cb_summary <- wr_cb_data %>%
  group_by(player_id, player_display_name, team, season) %>%
  summarise(
    interceptions = sum(def_interceptions, na.rm = TRUE),
    passes_defended = sum(def_pass_defended, na.rm = TRUE),
    solo_tackles = sum(def_tackles_solo, na.rm = TRUE),
    games_with_stats = n_distinct(game_id),
    .groups = "drop"
  ) %>%
  arrange(season, desc(passes_defended))

wr_cb_summary %>%
  filter(season == 2025) %>%
  arrange(desc(passes_defended)) %>%
  head(20)

wr_classifications <- bind_rows(

  tibble(
    player_name = c(
      "Ja'Marr Chase",
      "CeeDee Lamb",
      "George Pickens",
      "Amon-Ra St. Brown",
      "Puka Nacua",
      "Justin Jefferson",
      "Jaxon Smith-Njigba"
    ),
    wr_rating = "Elite"
  ),

  tibble(
    player_name = c(
      "Drake London",
      "Zay Flowers",
      "DJ Moore",
      "Khalil Shakir",
      "Tetairoa McMillan",
      "Jalen Coker",
      "Luther Burden III",
      "Rome Odunze",
      "Tee Higgins",
      "Denzel Boston",
      "KC Concepcion",
      "Jaylen Waddle",
      "Courtland Sutton",
      "Jameson Williams",
      "Christian Watson",
      "Matthew Golden",
      "Nico Collins",
      "Parker Washington",
      "Jakobi Meyers",
      "Rashee Rice",
      "Josh Downs",
      "Ladd McConkey",
      "Jordan Addison",
      "A.J. Brown",
      "Chris Olave",
      "Malik Nabers",
      "Garrett Wilson",
      "Omar Cooper Jr.",
      "DeVonta Smith",
      "Makai Lemon",
      "DK Metcalf",
      "Germie Bernard",
      "Mike Evans",
      "Deebo Samuel Sr.",
      "Keenan Allen",
      "Cooper Kupp",
      "Emeka Egbuka",
      "Chris Godwin Jr.",
      "Carnell Tate",
      "Stefon Diggs"
    ),
    wr_rating = "Good"
  ),

  tibble(
    player_name = c(
      "Reggie Virgil", "Simi Fehoko", "Devin Duvernay",
      "Jahan Dotson", "Olamide Zaccheaus",
      "Rashod Bateman", "Ja'Kobi Lane",
      "Keon Coleman", "John Metchie III",
      "Kalif Raymond", "Dohnte Meyers",
      "Jerry Jeudy", "Ryan Flournoy", "Pat Bryant",
      "Isaac TeSlaa", "Savion Williams",
      "Xavier Hutchinson", "Tank Dell",
      "Laquon Treadwell", "Ashton Dulin", "Anthony Gould",
      "Brian Thomas Jr.", "Xavier Worthy",
      "Tre Tucker", "Jalen Nailor", "Jack Bech",
      "Quentin Johnston", "Tre' Harris",
      "Konata Mumpfield", "Tutu Atwell",
      "Malik Washington", "Caleb Douglas", "Chris Bell",
      "Jauan Jennings", "DeMario Douglas", "Mack Hollins",
      "Devaughn Vele", "Jordyn Tyson",
      "Malachi Fields", "Darnell Mooney",
      "Adonai Mitchell", "Dontayvion Wicks",
      "Ben Skowronek", "Rashid Shaheed", "Ted Hurst III",
      "Wan'Dale Robinson", "Elic Ayomanor",
      "Antonio Williams", "Treylon Burks"
    ),
    wr_rating = "Other"
  )
)

nrow(wr_classifications)

wr_classifications %>%
  count(wr_rating)

wr_classifications %>%
  count(player_name) %>%
  filter(n > 1)

wr_player_ids <- wr_model_data %>%
  distinct(player_id, player_display_name)

wr_classifications_id <- wr_classifications %>%
  left_join(
    wr_player_ids,
    by = c("player_name" = "player_display_name")
  )

wr_classifications_id %>%
  filter(is.na(player_id))

wr_classifications_id %>%
  summarise(
    total_players = n(),
    matched_players = sum(!is.na(player_id)),
    unmatched_players = sum(is.na(player_id))
  )

wr_player_ids %>%
  filter(grepl("Harris", player_display_name, ignore.case = TRUE))

wr_classifications <- wr_classifications %>%
  mutate(
    player_name = recode(
      player_name,
      "Tre' Harris" = "Tre Harris"
    )
  )

wr_classifications_id <- wr_classifications %>%
  left_join(
    wr_player_ids,
    by = c("player_name" = "player_display_name")
  )

wr_classifications_id %>%
  filter(is.na(player_id))

# Joining the wrs classification to the wr model data
wr_model_data <- wr_model_data %>%
  left_join(
    wr_classifications_id %>%
      select(player_id, wr_rating),
    by = "player_id"
  )

wr_model_data %>%
  distinct(player_id, wr_rating) %>%
  count(wr_rating)

wr_model_data %>%
  filter(is.na(wr_rating)) %>%
  distinct(player_id, player_display_name)

wr_model_data %>%
  filter(player_display_name %in% c(
    "Justin Jefferson",
    "Ja'Marr Chase",
    "Puka Nacua",
    "Drake London"
  )) %>%
  select(player_display_name, season, week, wr_rating) %>%
  head(12)

wr_model_data %>%
  filter(is.na(wr_rating)) %>%
  distinct(player_id, player_display_name) %>%
  count(player_id) %>%
  filter(n > 1)

wr_model_data %>%
  filter(player_id == "00-0038740") %>%
  distinct(player_id, player_name, player_display_name)

# Cb classifications
cb_classifications <- bind_rows(

  # LOCKDOWN CORNERS
  tibble(
    player_name = c(
      "A.J. Terrell Jr.",
      "Kyle Hamilton",
      "Denzel Ward",
      "Pat Surtain II",
      "Derek Stingley Jr.",
      "Trent McDuffie",
      "Christian Gonzalez",
      "Quinyon Mitchell",
      "Cooper DeJean",
      "Jalen Ramsey",
      "Devon Witherspoon",
      "Nick Emmanwori",
      "Jaylen Watson",
      "Jaycee Horn",
      "Mike Jackson"
    ),
    cb_rating = "Lockdown"
  ),

  # GOOD CORNERS
  tibble(
    player_name = c(
      "Denzel Burke",
      "C.J. Henderson",
      "Nate Wiggins",
      "Marlon Humphrey",
      "Christian Benford",
      "Jaylon Johnson",
      "Tyrique Stevenson",
      "Kyler Gordon",
      "D.J. Turner II",
      "Joey Porter Jr.",
      "DaRon Bland",
      "Caleb Downs",
      "Riley Moss",
      "D.J. Reed",
      "Keisean Nixon",
      "Javon Bullard",
      "Kamari Lassiter",
      "Jalen Pitre",
      "Sauce Gardner",
      "Charvarius Ward",
      "Travis Hunter",
      "Jourdan Lewis",
      "Nohl Williams",
      "Mansoor Delane",
      "Josh Wallace",
      "Quentin Lake",
      "Donte Jackson",
      "Isaiah Rodgers",
      "Byron Murphy Jr.",
      "Kool-Aid McKinstry",
      "Greg Newsome II",
      "Riq Woolen",
      "Asante Samuel Jr.",
      "Jamel Dean",
      "Josh Jobe",
      "Renardo Green",
      "Deommodore Lenoir",
      "Upton Stout",
      "Zyon McCollum",
      "Rasul Douglas",
      "Amik Robertson",
      "Will Johnson",
      "L'Jarius Sneed"
    ),
    cb_rating = "Good"
  ),

  # OTHER CORNERS
  tibble(
    player_name = c(
      "Kalen King",
      "Max Melton",
      "Sydney Brown",
      "Davison Igbinosun",
      "Dee Alford",
      "Malcolm DeWalt IV",
      "Will Lee III",
      "Chau Smith-Wade",
      "Tacario Davis",
      "Dax Hill",
      "Myles Harden",
      "Ennis Rakestraw Jr.",
      "Roger McCreary",
      "Brandon Cisse",
      "Ja'Quan McMillian",
      "Cam Hart",
      "Tarheeb Still",
      "Eric Stokes",
      "Darien Porter",
      "Taron Johnson",
      "Jason Marshall Jr.",
      "JuJu Brents",
      "Chris Johnson",
      "James Pierre",
      "Martin Emerson Jr.",
      "Jonas Sanker",
      "Deonte Banks",
      "Dru Phillips",
      "Azareye'h Thomas",
      "Brandon Stephens",
      "Jarvis Brownlee Jr.",
      "Benjamin Morrison",
      "Jacob Parrish",
      "Joshua Williams",
      "Cor'Dale Flott",
      "Marcus Harris",
      "Trey Amos",
      "Bilhal Kone",
      "Akayleb Evans",
      "Dallis Flowers",
      "Jalen Davis",
      "Ja'Sir Taylor",
      "Damarri Mathis",
      "Devin Moore",
      "Jaylin Smith",
      "Darrell Baker",
      "Rejzohn Wright",
      "Thaddeus Dixon",
      "Jarrick Bernard-Converse",
      "Rico Payton",
      "Tre Brown",
      "Jakorian Bennett",
      "Michael Carter II",
      "Nate Hobbs"
    ),
    cb_rating = "Other"
  )
)

nrow(cb_classifications)

cb_classifications %>%
  count(cb_rating)

cb_classifications %>%
  count(player_name) %>%
  filter(n > 1)

wr_cb_data %>%
  filter(player_display_name %in% c(
    "Pat Surtain II",
    "Jalen Ramsey",
    "Denzel Ward",
    "Trent McDuffie"
  )) %>%
  distinct(
    player_id,
    player_display_name,
    season,
    team
  ) %>%
  arrange(player_display_name, season)

cb_player_ids <- wr_cb_data %>%
  distinct(player_id, player_display_name)

cb_classifications_id <- cb_classifications %>%
  left_join(
    cb_player_ids,
    by = c("player_name" = "player_display_name")
  )

cb_classifications_id %>%
  filter(is.na(player_id))

cb_classifications_id %>%
  summarise(
    total_players = n(),
    matched_players = sum(!is.na(player_id)),
    unmatched_players = sum(is.na(player_id))
  )

all_player_ids <- load_player_stats(
  seasons = 2020:2026,
  summary_level = "week"
) %>%
  distinct(
    player_id,
    player_display_name,
    position,
    position_group
  )

cb_unmatched <- cb_classifications_id %>%
  filter(is.na(player_id)) %>%
  select(player_name, cb_rating)

cb_unmatched_check <- cb_unmatched %>%
  left_join(
    all_player_ids,
    by = c("player_name" = "player_display_name")
  )

cb_unmatched_check %>%
  arrange(player_name)

print(cb_unmatched_check, n = Inf)

all_nfl_players <- nflreadr::load_players()

names(all_nfl_players)

cb_master_lookup <- all_nfl_players %>%
  select(
    player_id = gsis_id,
    player_name = display_name,
    position,
    position_group,
    latest_team,
    pff_position,
    pff_status
  )

cb_classifications_master <- cb_classifications %>%
  left_join(
    cb_master_lookup,
    by = "player_name"
  )

cb_classifications_master %>%
  summarise(
    total_players = n(),
    matched_players = sum(!is.na(player_id)),
    unmatched_players = sum(is.na(player_id))
  )

cb_classifications_master %>%
  filter(is.na(player_id)) %>%
  select(player_name, cb_rating)

cb_classifications_master %>%
  select(
    player_name,
    cb_rating,
    position,
    position_group,
    latest_team,
    pff_position,
    pff_status
  ) %>%
  arrange(player_name)

cb_classifications_master %>%
  count(pff_position, sort = TRUE)

cb_classifications_master %>%
  count(pff_status, sort = TRUE)

cb_master_lookup %>%
  semi_join(
    cb_classifications,
    by = c("player_name" = "player_name")
  ) %>%
  count(player_name, sort = TRUE) %>%
  filter(n > 1)

all_nfl_players %>%
  filter(
    grepl(
      "Terrell|Turner|Murphy|Hill|Brents|Emerson|Phillips|Flott|Baker",
      display_name,
      ignore.case = TRUE
    )
  ) %>%
  select(
    gsis_id,
    display_name,
    first_name,
    last_name,
    football_name,
    position,
    position_group,
    latest_team,
    pff_position,
    pff_status
  ) %>%
  arrange(last_name, first_name)

cb_possible_matches <- all_nfl_players %>%
  filter(
    last_name %in% c(
      "Terrell",
      "Turner",
      "Murphy",
      "Hill",
      "Brents",
      "Emerson",
      "Phillips",
      "Flott",
      "Baker"
    ),
    position_group == "DB"
  ) %>%
  select(
    gsis_id,
    display_name,
    first_name,
    last_name,
    football_name,
    position,
    latest_team
  ) %>%
  arrange(last_name, first_name)

print(cb_possible_matches, n = Inf)

all_nfl_players %>%
  filter(display_name %in% c(
    "Chris Johnson",
    "Eric Stokes",
    "Marcus Harris",
    "Will Johnson",
    "Brandon Stephens",
    "Devin Moore",
    "Michael Jackson"
  )) %>%
  select(
    gsis_id,
    display_name,
    birth_date,
    position,
    latest_team,
    rookie_season,
    last_season
  ) %>%
  arrange(display_name, birth_date)

all_nfl_players %>%
  filter(display_name %in% c(
    "A.J. Terrell",
    "DJ Turner II",
    "Byron Murphy",
    "Daxton Hill",
    "Julius Brents",
    "Martin Emerson",
    "Andru Phillips",
    "Cordale Flott",
    "Darrell Baker Jr."
  )) %>%
  select(
    gsis_id,
    display_name,
    position,
    latest_team,
    rookie_season,
    last_season
  ) %>%
  arrange(display_name)

all_nfl_players %>%
  filter(display_name %in% c(
    "Chris Johnson",
    "Eric Stokes",
    "Marcus Harris",
    "Will Johnson",
    "Brandon Stephens",
    "Devin Moore",
    "Michael Jackson"
  )) %>%
  select(
    gsis_id,
    display_name,
    birth_date,
    position,
    position_group,
    latest_team,
    rookie_season,
    last_season
  ) %>%
  arrange(display_name, birth_date)

cb_master_lookup_2026 <- all_nfl_players %>%
  filter(last_season == 2026) %>%
  select(
    player_id = gsis_id,
    player_name = display_name,
    position,
    position_group,
    latest_team
  )

cb_master_lookup_2026 %>%
  count(player_name) %>%
  filter(n > 1)

cb_master_lookup_2026 %>%
  filter(player_name %in% c(
    "Brandon Johnson",
    "Byron Young",
    "Jaylon Jones",
    "Justin Jefferson",
    "Marcus Harris"
  )) %>%
  arrange(player_name, latest_team) %>%
  print(n = Inf)

cb_master_lookup_2026 <- all_nfl_players %>%
  filter(
    last_season == 2026,
    position_group == "DB"
  ) %>%
  select(
    player_id = gsis_id,
    player_name = display_name,
    position,
    position_group,
    latest_team
  )

cb_master_lookup_2026 %>%
  count(player_name) %>%
  filter(n > 1)

cb_classifications_id <- cb_classifications %>%
  left_join(
    cb_master_lookup_2026,
    by = "player_name"
  )

cb_classifications_id %>%
  summarise(
    total_rows = n(),
    matched = sum(!is.na(player_id)),
    unmatched = sum(is.na(player_id)),
    unique_players = n_distinct(player_id, na.rm = TRUE)
  )

cb_classifications_id %>%
  filter(is.na(player_id)) %>%
  select(player_name, cb_rating)

cb_name_map <- tibble(
  player_name = c(
    "A.J. Terrell Jr.",
    "Mike Jackson",
    "D.J. Turner II",
    "Byron Murphy Jr.",
    "Dax Hill",
    "JuJu Brents",
    "Martin Emerson Jr.",
    "Dru Phillips",
    "Cor'Dale Flott",
    "Darrell Baker"
  ),
  lookup_name = c(
    "A.J. Terrell",
    "Mike Jackson",
    "DJ Turner II",
    "Byron Murphy",
    "Daxton Hill",
    "Julius Brents",
    "Martin Emerson",
    "Andru Phillips",
    "Cordale Flott",
    "Darrell Baker Jr."
  )
)

cb_classifications_id <- cb_classifications %>%
  left_join(cb_name_map, by = "player_name") %>%
  mutate(lookup_name = coalesce(lookup_name, player_name)) %>%
  left_join(
    cb_master_lookup_2026,
    by = c("lookup_name" = "player_name")
  ) %>%
  select(-lookup_name)

cb_classifications_id %>%
  summarise(
    total_rows = n(),
    matched = sum(!is.na(player_id)),
    unmatched = sum(is.na(player_id)),
    unique_players = n_distinct(player_id, na.rm = TRUE)
  )

cb_classifications_id %>%
  filter(is.na(player_id)) %>%
  select(player_name, cb_rating)

wr_db_data <- load_player_stats(
  seasons = 2020:2026,
  summary_level = "week"
) %>%
  filter(
    season_type == "REG",
    position_group == "DB"
  )

dim(wr_db_data)

wr_db_data %>%
  count(position)

wr_db_data <- wr_db_data %>%
  inner_join(
    cb_classifications_id %>%
      select(player_id, cb_rating),
    by = "player_id"
  )

wr_db_data %>%
  summarise(
    total_rows = n(),
    unique_players = n_distinct(player_id),
    missing_ratings = sum(is.na(cb_rating))
  )

wr_db_data %>%
  count(cb_rating)

cb_classifications_id %>%
  anti_join(
    wr_db_data %>% distinct(player_id),
    by = "player_id"
  ) %>%
  select(player_name, cb_rating, position, latest_team)

names(wr_db_data)

wr_db_matchups <- wr_db_data %>%
  group_by(
    game_id,
    season,
    week,
    defense_team = team
  ) %>%
  summarise(
    lockdown_db_count = sum(cb_rating == "Lockdown"),
    good_db_count = sum(cb_rating == "Good"),
    other_db_count = sum(cb_rating == "Other"),
    classified_db_count = n_distinct(player_id),
    .groups = "drop"
  )

dim(wr_db_matchups)

wr_db_matchups %>%
  arrange(season, week, defense_team) %>%
  print(n = 10)

wr_model_data <- wr_model_data %>%
  left_join(
    wr_db_matchups,
    by = c(
      "game_id",
      "season",
      "week",
      "opponent_team" = "defense_team"
    )
  )

wr_model_data %>%
  summarise(
    total_wr_games = n(),
    matched_games = sum(!is.na(classified_db_count)),
    unmatched_games = sum(is.na(classified_db_count)),
    lockdown_records = sum(lockdown_db_count, na.rm = TRUE),
    good_records = sum(good_db_count, na.rm = TRUE),
    other_records = sum(other_db_count, na.rm = TRUE)
  )

wr_model_data %>%
  filter(is.na(classified_db_count)) %>%
  select(
    player_name,
    season,
    week,
    team,
    opponent_team,
    game_id
  ) %>%
  distinct() %>%
  arrange(season, week) %>%
  print(n = 15)

wr_model_data %>%
  filter(season == 2026) %>%
  summarise(
    total_wr_games = n(),
    matched_games = sum(!is.na(classified_db_count)),
    unmatched_games = sum(is.na(classified_db_count))
  )

wr_train_data <- wr_model_data %>%
  filter(season <= 2025)

wr_2026_data <- wr_model_data %>%
  filter(season == 2026)

dim(wr_train_data)
dim(wr_2026_data)

wr_2026_data %>%
  count(week, name = "wr_game_records") %>%
  arrange(week)

wr_2026_data %>%
  group_by(week) %>%
  summarise(
    total_games = n(),
    matched_games = sum(!is.na(classified_db_count)),
    unmatched_games = sum(is.na(classified_db_count)),
    .groups = "drop"
  ) %>%
  arrange(week)

wr_2026_matched <- wr_2026_data %>%
  filter(!is.na(classified_db_count))

wr_2026_train <- wr_2026_matched %>%
  filter(week <= 7)

wr_2026_validation <- wr_2026_matched %>%
  filter(week >= 8)

dim(wr_2026_matched)
dim(wr_2026_train)
dim(wr_2026_validation)

wr_2026_train <- wr_2026_train %>%
  mutate(
    any_receiving_td = as.integer(receiving_tds > 0)
  )

wr_2026_validation <- wr_2026_validation %>%
  mutate(
    any_receiving_td = as.integer(receiving_tds > 0)
  )

wr_2026_train %>%
  summarise(
    avg_receptions = mean(receptions, na.rm = TRUE),
    avg_receiving_yards = mean(receiving_yards, na.rm = TRUE),
    avg_receiving_tds = mean(receiving_tds, na.rm = TRUE),
    games_with_td = sum(any_receiving_td),
    games_without_td = sum(any_receiving_td == 0)
  )

wr_2026_train %>%
  summarise(
    avg_targets = mean(targets, na.rm = TRUE),
    avg_receptions = mean(receptions, na.rm = TRUE),
    avg_receiving_yards = mean(receiving_yards, na.rm = TRUE),
    avg_rushing_yards = mean(rushing_yards, na.rm = TRUE)
  )

names(wr_2026_train)

wr_2026_train <- wr_2026_train %>%
  select(-matches("\\.(x|y)$"))

wr_2026_validation <- wr_2026_validation %>%
  select(-matches("\\.(x|y)$"))

names(wr_2026_train)
dim(wr_2026_train)

names(wr_2026_validation)
dim(wr_2026_validation)

wr_predictors <- c(
  # WR historical performance
  "prev_receiving_yards",
  "prev_receiving_tds",
  "prev_targets",
  "prev_receptions",
  "prev_rushing_yards",
  "prev_carries",
  "avg_receiving_yards_3",
  "avg_receiving_tds_3",
  "avg_targets_3",
  "avg_receptions_3",
  "avg_rushing_yards_3",
  "avg_carries_3",
  "avg_receiving_yards_5",
  "avg_receiving_tds_5",
  "avg_targets_5",
  "avg_receptions_5",
  "avg_rushing_yards_5",
  "avg_carries_5",
  "prev_td_indicator",
  "td_streak",
  "td_rate_last_3",
  "td_rate_last_5",

  # Historical NGS metrics
  grep("^ngs_", names(wr_2026_train), value = TRUE),

  # Historical opponent defense
  grep(
    "^(prev_|avg3_|avg5_).*(allowed|made|recorded)$",
    names(wr_2026_train),
    value = TRUE
  ),

  # Player and opposing DB ratings
  "wr_rating",
  "lockdown_db_count",
  "good_db_count",
  "other_db_count"
)

length(wr_predictors)
setdiff(wr_predictors, names(wr_2026_train))

wr_missing <- wr_2026_train %>%
  summarise(
    across(
      all_of(wr_predictors),
      ~ sum(is.na(.))
    )
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = "predictor",
    values_to = "missing"
  ) %>%
  filter(missing > 0) %>%
  arrange(desc(missing))

wr_missing

sum(is.na(wr_2026_train[wr_predictors]))

print(wr_missing, n = 40)

wr_predictors <- setdiff(
  wr_predictors,
  c(
    "wr_rating",
    "lockdown_db_count",
    "good_db_count",
    "other_db_count"
  )
)

wr_train_data <- wr_model_data %>%
  filter(season >= 2020, season <= 2024)

wr_validation_data <- wr_model_data %>%
  filter(season == 2025)

wr_2026_data <- wr_model_data %>%
  filter(season == 2026)

dim(wr_train_data)
dim(wr_validation_data)
dim(wr_2026_data)

wr_predictors <- c(
  "prev_receiving_yards",
  "prev_receiving_tds",
  "prev_targets",
  "prev_receptions",
  "prev_rushing_yards",
  "prev_carries",
  "avg_receiving_yards_3",
  "avg_receiving_tds_3",
  "avg_targets_3",
  "avg_receptions_3",
  "avg_rushing_yards_3",
  "avg_carries_3",
  "avg_receiving_yards_5",
  "avg_receiving_tds_5",
  "avg_targets_5",
  "avg_receptions_5",
  "avg_rushing_yards_5",
  "avg_carries_5",
  "prev_td_indicator",
  "td_streak",
  "td_rate_last_3",
  "td_rate_last_5",
  grep("^ngs_", names(wr_train_data), value = TRUE),
  grep(
    "^(prev_|avg3_|avg5_).*(allowed|made|recorded)$",
    names(wr_train_data),
    value = TRUE
  )
)

length(wr_predictors)
setdiff(wr_predictors, names(wr_train_data))

wr_missing <- wr_train_data %>%
  summarise(
    across(
      all_of(wr_predictors),
      ~ sum(is.na(.))
    )
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = "predictor",
    values_to = "missing"
  ) %>%
  filter(missing > 0) %>%
  arrange(desc(missing))

print(wr_missing, n = 73)

sum(is.na(wr_train_data[wr_predictors]))

wr_train_data %>%
  summarise(
    total_rows = n(),
    rows_with_ngs = sum(
      if_any(starts_with("ngs_"), ~ !is.na(.))
    ),
    rows_without_ngs = sum(
      if_all(starts_with("ngs_"), is.na)
    )
  )

wr_missing %>%
  filter(missing == nrow(wr_train_data))

wr_train_data <- wr_train_data %>%
  mutate(any_receiving_td = as.integer(receiving_tds > 0))

wr_validation_data <- wr_validation_data %>%
  mutate(any_receiving_td = as.integer(receiving_tds > 0))

wr_2026_data <- wr_2026_data %>%
  mutate(any_receiving_td = as.integer(receiving_tds > 0))

wr_train_data %>%
  summarise(
    avg_targets = mean(targets, na.rm = TRUE),
    avg_receptions = mean(receptions, na.rm = TRUE),
    avg_receiving_yards = mean(receiving_yards, na.rm = TRUE),
    games_with_td = sum(any_receiving_td),
    games_without_td = sum(any_receiving_td == 0)
  )

wr_train_data %>%
  summarise(
    missing_targets = sum(is.na(targets)),
    missing_receptions = sum(is.na(receptions)),
    missing_receiving_yards = sum(is.na(receiving_yards)),
    missing_td_target = sum(is.na(any_receiving_td))
  )

wr_recipe <- recipe(
  ~ .,
  data = wr_train_data %>%
    select(all_of(wr_predictors))
) %>%
  step_indicate_na(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

wr_recipe

wr_recipe_prep <- prep(
  wr_recipe,
  training = wr_train_data %>%
    select(all_of(wr_predictors))
)

wr_train_processed <- bake(
  wr_recipe_prep,
  new_data = wr_train_data %>%
    select(all_of(wr_predictors))
)

dim(wr_train_processed)
sum(is.na(wr_train_processed))

wr_validation_processed <- bake(
  wr_recipe_prep,
  new_data = wr_validation_data %>%
    select(all_of(wr_predictors))
)

wr_2026_processed <- bake(
  wr_recipe_prep,
  new_data = wr_2026_data %>%
    select(all_of(wr_predictors))
)

dim(wr_train_processed)
dim(wr_validation_processed)
dim(wr_2026_processed)

sum(is.na(wr_validation_processed))
sum(is.na(wr_2026_processed))

wr_train_targets <- wr_train_data %>%
  select(
    targets,
    receptions,
    receiving_yards,
    any_receiving_td
  )

wr_validation_targets <- wr_validation_data %>%
  select(
    targets,
    receptions,
    receiving_yards,
    any_receiving_td
  )

wr_2026_targets <- wr_2026_data %>%
  select(
    targets,
    receptions,
    receiving_yards,
    any_receiving_td
  )

dim(wr_train_targets)
dim(wr_validation_targets)
dim(wr_2026_targets)

str(wr_train_targets)
str(wr_validation_targets)
str(wr_2026_targets)

regression_targets <- c(
  "targets",
  "receptions",
  "receiving_yards"
)

wr_target_means <- sapply(
  wr_train_targets[regression_targets],
  mean
)

wr_target_sds <- sapply(
  wr_train_targets[regression_targets],
  sd
)

wr_target_means
wr_target_sds

wr_train_targets_scaled <- wr_train_targets %>%
  mutate(
    across(
      all_of(regression_targets),
      ~ (.x - wr_target_means[cur_column()]) /
        wr_target_sds[cur_column()]
    )
  )

wr_validation_targets_scaled <- wr_validation_targets %>%
  mutate(
    across(
      all_of(regression_targets),
      ~ (.x - wr_target_means[cur_column()]) /
        wr_target_sds[cur_column()]
    )
  )

wr_2026_targets_scaled <- wr_2026_targets %>%
  mutate(
    across(
      all_of(regression_targets),
      ~ (.x - wr_target_means[cur_column()]) /
        wr_target_sds[cur_column()]
    )
  )

summary(wr_train_targets_scaled)
summary(wr_validation_targets_scaled)
summary(wr_2026_targets_scaled)

wr_x_train <- torch_tensor(
  as.matrix(wr_train_processed),
  dtype = torch_float()
)

wr_x_validation <- torch_tensor(
  as.matrix(wr_validation_processed),
  dtype = torch_float()
)

wr_x_2026 <- torch_tensor(
  as.matrix(wr_2026_processed),
  dtype = torch_float()
)

wr_y_train <- torch_tensor(
  as.matrix(wr_train_targets_scaled),
  dtype = torch_float()
)

wr_y_validation <- torch_tensor(
  as.matrix(wr_validation_targets_scaled),
  dtype = torch_float()
)

wr_y_2026 <- torch_tensor(
  as.matrix(wr_2026_targets_scaled),
  dtype = torch_float()
)

wr_x_train$shape
wr_x_validation$shape
wr_x_2026$shape

wr_y_train$shape
wr_y_validation$shape
wr_y_2026$shape

wr_model <- nn_module(
  "WRModel",

  initialize = function() {
    self$fc1 <- nn_linear(145, 128)
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

wr_net <- wr_model()

print(wr_net)

wr_test_output <- wr_net(wr_x_train)

wr_test_output$targets$shape
wr_test_output$receptions$shape
wr_test_output$yards$shape
wr_test_output$td$shape

wr_mse_loss <- nn_mse_loss()

wr_td_loss <- nn_bce_with_logits_loss()

wr_optimizer <- optim_adam(
  wr_net$parameters,
  lr = 0.001
)

wr_combined_loss <- function(predictions, targets) {

  targets_loss <- wr_mse_loss(
    predictions$targets,
    targets$narrow(2, 1, 1)
  )

  receptions_loss <- wr_mse_loss(
    predictions$receptions,
    targets$narrow(2, 2, 1)
  )

  yards_loss <- wr_mse_loss(
    predictions$yards,
    targets$narrow(2, 3, 1)
  )

  td_loss <- wr_td_loss(
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

wr_test_losses <- wr_combined_loss(
  wr_test_output,
  wr_y_train
)

wr_test_losses$total$item()
wr_test_losses$targets$item()
wr_test_losses$receptions$item()
wr_test_losses$yards$item()
wr_test_losses$td$item()

wr_epochs <- 30

wr_training_losses <- numeric(wr_epochs)

for (epoch in 1:wr_epochs) {

  wr_net$train()

  wr_predictions <- wr_net(wr_x_train)

  wr_losses <- wr_combined_loss(
    wr_predictions,
    wr_y_train
  )

  wr_optimizer$zero_grad()

  wr_losses$total$backward()

  wr_optimizer$step()

  wr_training_losses[epoch] <- wr_losses$total$item()

  cat(
    "Epoch:", epoch,
    "| Loss:", round(wr_training_losses[epoch], 4),
    "\n"
  )
}

wr_net$eval()

with_no_grad({

  wr_validation_predictions <- wr_net(wr_x_validation)

  wr_validation_losses <- wr_combined_loss(
    wr_validation_predictions,
    wr_y_validation
  )

})

wr_validation_losses$total$item()
wr_validation_losses$targets$item()
wr_validation_losses$receptions$item()
wr_validation_losses$yards$item()
wr_validation_losses$td$item()

wr_validation_results <- tibble(
  actual_targets = wr_validation_targets$targets,
  predicted_targets = as.numeric(
    as_array(wr_validation_predictions$targets)
  ) * wr_target_sds["targets"] +
    wr_target_means["targets"],

  actual_receptions = wr_validation_targets$receptions,
  predicted_receptions = as.numeric(
    as_array(wr_validation_predictions$receptions)
  ) * wr_target_sds["receptions"] +
    wr_target_means["receptions"],

  actual_yards = wr_validation_targets$receiving_yards,
  predicted_yards = as.numeric(
    as_array(wr_validation_predictions$yards)
  ) * wr_target_sds["receiving_yards"] +
    wr_target_means["receiving_yards"]
)

wr_validation_metrics <- tibble(
  target = c("Targets", "Receptions", "Receiving Yards"),

  MAE = c(
    mean(abs(wr_validation_results$actual_targets -
             wr_validation_results$predicted_targets)),
    mean(abs(wr_validation_results$actual_receptions -
             wr_validation_results$predicted_receptions)),
    mean(abs(wr_validation_results$actual_yards -
             wr_validation_results$predicted_yards))
  ),

  RMSE = c(
    sqrt(mean((wr_validation_results$actual_targets -
               wr_validation_results$predicted_targets)^2)),
    sqrt(mean((wr_validation_results$actual_receptions -
               wr_validation_results$predicted_receptions)^2)),
    sqrt(mean((wr_validation_results$actual_yards -
               wr_validation_results$predicted_yards)^2))
  )
)

wr_validation_metrics

wr_td_probabilities <- torch_sigmoid(
  wr_validation_predictions$td
)

wr_td_probabilities <- as.numeric(
  as_array(wr_td_probabilities)
)

wr_td_actual <- wr_validation_targets$any_receiving_td

wr_td_predictions <- ifelse(
  wr_td_probabilities >= 0.5,
  1,
  0
)

wr_td_confusion <- table(
  Actual = wr_td_actual,
  Predicted = wr_td_predictions
)

wr_td_confusion

wr_td_accuracy <- mean(
  wr_td_predictions == wr_td_actual
)

wr_td_precision <- sum(
  wr_td_predictions == 1 & wr_td_actual == 1
) / max(sum(wr_td_predictions == 1), 1)

wr_td_recall <- sum(
  wr_td_predictions == 1 & wr_td_actual == 1
) / max(sum(wr_td_actual == 1), 1)

wr_td_f1 <- 2 * (
  wr_td_precision * wr_td_recall
) / max(wr_td_precision + wr_td_recall, 1e-8)

wr_td_brier <- mean(
  (wr_td_probabilities - wr_td_actual)^2
)

c(
  Accuracy = wr_td_accuracy,
  Precision = wr_td_precision,
  Recall = wr_td_recall,
  F1 = wr_td_f1,
  Brier = wr_td_brier
)

summary(wr_td_probabilities)

quantile(
  wr_td_probabilities,
  probs = c(0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1)
)

mean(wr_td_actual)

wr_thresholds <- seq(0.10, 0.40, by = 0.05)

wr_threshold_results <- lapply(
  wr_thresholds,
  function(threshold) {

    predictions <- ifelse(
      wr_td_probabilities >= threshold,
      1,
      0
    )

    tp <- sum(predictions == 1 & wr_td_actual == 1)
    fp <- sum(predictions == 1 & wr_td_actual == 0)
    fn <- sum(predictions == 0 & wr_td_actual == 1)
    tn <- sum(predictions == 0 & wr_td_actual == 0)

    precision <- tp / max(tp + fp, 1)
    recall <- tp / max(tp + fn, 1)

    f1 <- 2 * precision * recall /
      max(precision + recall, 1e-8)

    tibble(
      threshold = threshold,
      predicted_TDs = sum(predictions),
      TP = tp,
      FP = fp,
      FN = fn,
      TN = tn,
      accuracy = (tp + tn) / length(predictions),
      precision = precision,
      recall = recall,
      F1 = f1
    )
  }
)

wr_threshold_results <- bind_rows(wr_threshold_results)

wr_threshold_results

wr_td_auc_data <- tibble(
  actual_td = factor(
    wr_td_actual,
    levels = c(0, 1)
  ),
  td_probability = wr_td_probabilities
)

wr_td_auc <- roc_auc(
  wr_td_auc_data,
  truth = actual_td,
  td_probability,
  event_level = "second"
)

wr_td_auc

saveRDS(
  wr_net,
  "wr_model_baseline.rds"
)

saveRDS(
  wr_recipe_prep,
  "wr_recipe_baseline.rds"
)

saveRDS(
  wr_target_means,
  "wr_target_means.rds"
)

saveRDS(
  wr_target_sds,
  "wr_target_sds.rds"
)

wr_net$eval()

with_no_grad({

  wr_2026_predictions <- wr_net(wr_x_2026)

  wr_2026_losses <- wr_combined_loss(
    wr_2026_predictions,
    wr_y_2026
  )

})

cat("2026 Total Loss:",
    wr_2026_losses$total$item(), "\n")

cat("Targets Loss:",
    wr_2026_losses$targets$item(), "\n")

cat("Receptions Loss:",
    wr_2026_losses$receptions$item(), "\n")

cat("Receiving Yards Loss:",
    wr_2026_losses$yards$item(), "\n")

cat("Touchdown Loss:",
    wr_2026_losses$td$item(), "\n")

wr_2026_results <- tibble(
  actual_targets = wr_2026_targets$targets,
  predicted_targets = as.numeric(
    as_array(wr_2026_predictions$targets)
  ) * wr_target_sds["targets"] +
    wr_target_means["targets"],

  actual_receptions = wr_2026_targets$receptions,
  predicted_receptions = as.numeric(
    as_array(wr_2026_predictions$receptions)
  ) * wr_target_sds["receptions"] +
    wr_target_means["receptions"],

  actual_yards = wr_2026_targets$receiving_yards,
  predicted_yards = as.numeric(
    as_array(wr_2026_predictions$yards)
  ) * wr_target_sds["receiving_yards"] +
    wr_target_means["receiving_yards"]
)

wr_2026_metrics <- tibble(
  target = c("Targets", "Receptions", "Receiving Yards"),

  MAE = c(
    mean(abs(wr_2026_results$actual_targets -
             wr_2026_results$predicted_targets)),
    mean(abs(wr_2026_results$actual_receptions -
             wr_2026_results$predicted_receptions)),
    mean(abs(wr_2026_results$actual_yards -
             wr_2026_results$predicted_yards))
  ),

  RMSE = c(
    sqrt(mean((wr_2026_results$actual_targets -
               wr_2026_results$predicted_targets)^2)),
    sqrt(mean((wr_2026_results$actual_receptions -
               wr_2026_results$predicted_receptions)^2)),
    sqrt(mean((wr_2026_results$actual_yards -
               wr_2026_results$predicted_yards)^2))
  )
)

wr_2026_metrics

wr_2026_recent_baseline <- tibble(
  target = c("Targets", "Receptions", "Receiving Yards"),

  MAE = c(
    mean(abs(
      wr_2026_data$targets -
        wr_2026_data$avg_targets_3
    ), na.rm = TRUE),

    mean(abs(
      wr_2026_data$receptions -
        wr_2026_data$avg_receptions_3
    ), na.rm = TRUE),

    mean(abs(
      wr_2026_data$receiving_yards -
        wr_2026_data$avg_receiving_yards_3
    ), na.rm = TRUE)
  ),

  RMSE = c(
    sqrt(mean(
      (wr_2026_data$targets -
         wr_2026_data$avg_targets_3)^2,
      na.rm = TRUE
    )),

    sqrt(mean(
      (wr_2026_data$receptions -
         wr_2026_data$avg_receptions_3)^2,
      na.rm = TRUE
    )),

    sqrt(mean(
      (wr_2026_data$receiving_yards -
         wr_2026_data$avg_receiving_yards_3)^2,
      na.rm = TRUE
    ))
  )
)

wr_2026_recent_baseline

wr_2026_comparison <- wr_2026_metrics %>%
  rename(
    Neural_Network_MAE = MAE,
    Neural_Network_RMSE = RMSE
  ) %>%
  left_join(
    wr_2026_recent_baseline %>%
      rename(
        Recent_Average_MAE = MAE,
        Recent_Average_RMSE = RMSE
      ),
    by = "target"
  )

wr_2026_comparison

cat("2026 WR NEURAL NETWORK RESULTS\n")
cat("--------------------------------\n")

print(wr_2026_metrics)

cat("\n2026 MODEL LOSSES\n")
cat("--------------------------------\n")

cat("Total Loss:", wr_2026_losses$total$item(), "\n")
cat("Targets Loss:", wr_2026_losses$targets$item(), "\n")
cat("Receptions Loss:", wr_2026_losses$receptions$item(), "\n")
cat("Receiving Yards Loss:", wr_2026_losses$yards$item(), "\n")
cat("Touchdown Loss:", wr_2026_losses$td$item(), "\n")

cat("\nTOUCHDOWN CLASSIFICATION AUC\n")
print(wr_td_auc)

wr_2026_prediction_results <- wr_2026_data %>%
  select(
    player_id,
    player_display_name,
    team,
    opponent_team,
    season,
    week,
    game_id,
    targets,
    receptions,
    receiving_yards,
    receiving_tds
  ) %>%
  rename(
    actual_targets = targets,
    actual_receptions = receptions,
    actual_receiving_yards = receiving_yards,
    actual_receiving_tds = receiving_tds
  ) %>%
  bind_cols(wr_2026_results %>%
    select(
      predicted_targets,
      predicted_receptions,
      predicted_yards
    )) %>%
  mutate(
    td_probability = as.numeric(
      as_array(torch_sigmoid(wr_2026_predictions$td))
    )
  ) %>%
  mutate(
    predicted_any_td = ifelse(td_probability >= 0.25, 1, 0)
  ) %>%
  select(
    player_id,
    player_display_name,
    team,
    opponent_team,
    season,
    week,
    game_id,
    actual_targets,
    predicted_targets,
    actual_receptions,
    predicted_receptions,
    actual_receiving_yards,
    predicted_yards,
    actual_receiving_tds,
    td_probability,
    predicted_any_td
  )

print(
  wr_2026_prediction_results,
  n = 25
)

View(wr_2026_prediction_results)
write.csv(
  wr_2026_prediction_results,
  "WR_2026_Neural_Network_Predictions.csv",
  row.names = FALSE
)

wr_2026_overall_errors <- tibble(
  targets_error =
    (wr_2026_results$actual_targets -
       wr_2026_results$predicted_targets) /
    wr_target_sds["targets"],

  receptions_error =
    (wr_2026_results$actual_receptions -
       wr_2026_results$predicted_receptions) /
    wr_target_sds["receptions"],

  yards_error =
    (wr_2026_results$actual_yards -
       wr_2026_results$predicted_yards) /
    wr_target_sds["receiving_yards"]
)

wr_2026_overall_metrics <- tibble(
  Metric = c("Overall MAE", "Overall RMSE"),
  Value = c(
    mean(abs(as.matrix(wr_2026_overall_errors))),
    sqrt(mean(as.matrix(wr_2026_overall_errors)^2))
  )
)

wr_2026_overall_metrics
