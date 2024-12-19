import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rpy2.robjects import pandas2ri, r
from IPython.display import Image, display
import os
from rpy2.robjects.packages import importr
pandas2ri.activate()
base = importr('base')
rpart = importr('rpart')
rpart_plot = importr('rpart.plot')
rattle = importr("rattle")
grdevices = importr("grDevices")
match_data =pd.read_csv('match_data/match_data.csv')

first_half_data = match_data[(match_data['halftime'] == '1st-half') & (match_data['suspended'] == False) & (match_data['stopped'] == False)]
second_half_data = match_data[(match_data['halftime'] == '2nd-half') & (match_data['suspended'] == False) & (match_data['stopped'] == False)]

def calculate_probabilities(df):
    df = df.copy()
    df['P_home'] = 1 / df['1']
    df['P_draw'] = 1 / df['X']
    df['P_away'] = 1 / df['2']
    total = df['P_home'] + df['P_draw'] + df['P_away']
    df['P_home_norm'] = df['P_home'] / total
    df['P_draw_norm'] = df['P_draw'] / total
    df['P_away_norm'] = df['P_away'] / total
    df['P_diff'] = df['P_home_norm'] - df['P_away_norm']
    return df

first_half_processed = calculate_probabilities(first_half_data)
second_half_processed = calculate_probabilities(second_half_data)

def bin_and_calculate_actual(df, bins):
    df['P_diff_bin'] = pd.cut(df['P_diff'], bins)
    total_counts = df.groupby('P_diff_bin')['result'].count()
    draw_counts = df[df['result'] == 'X'].groupby('P_diff_bin')['result'].count()
    actual_prob = draw_counts / total_counts
    return actual_prob

bins = np.arange(-1, 1.1, 0.01)
bin_midpoints = bins[:-1] + (bins[1] - bins[0]) / 2
first_half_actual = bin_and_calculate_actual(first_half_processed, bins)
second_half_actual = bin_and_calculate_actual(second_half_processed, bins)

def plot_draw(df, bin_midpoints, actual):
    actual_values = actual.values
    plt.figure(figsize=(10, 6))
    plt.scatter(df['P_diff'], df['P_draw_norm'], alpha=0.6, label="P(draw) by the bookmaker")
    plt.scatter(bin_midpoints, actual_values, color='red', marker='o', label="Actual P(draw)")
    if df['halftime'].iloc[0] == '1st-half':
        plt.title("P(draw) vs P(home)-P(away) for the first half", fontsize=14)
    else:
        plt.title("P(draw) vs P(home)-P(away) for the second half", fontsize=14)
    plt.xlabel("P(home)-P(away)", fontsize=12)
    plt.ylabel("P(draw)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.show()

plot_draw(first_half_processed,bin_midpoints,first_half_actual)
plot_draw(second_half_processed,bin_midpoints,second_half_actual)


bins = np.arange(-1, 1.1, 0.12)
bin_midpoints = bins[:-1] + (bins[1] - bins[0]) / 2
first_half_data_initial_bet = first_half_data.groupby('fixture_id').first()
second_half_data_initial_bet = second_half_data.groupby('fixture_id').first()
first_half_data_initial_bet_processed = calculate_probabilities(first_half_data_initial_bet)
second_half_data_initial_bet_processed = calculate_probabilities(second_half_data_initial_bet)
first_half_data_initial_bet_actual = bin_and_calculate_actual(first_half_data_initial_bet_processed, bins)
second_half_data_initial_bet_actual = bin_and_calculate_actual(second_half_data_initial_bet_processed, bins)
plot_draw(first_half_data_initial_bet_processed,bin_midpoints,first_half_data_initial_bet_actual)
plot_draw(second_half_data_initial_bet_processed,bin_midpoints,second_half_data_initial_bet_actual)

FH_data = match_data[(match_data['halftime'] == '1st-half') & (match_data['suspended'] == False) & (match_data['stopped'] == False)]
matches_with_early_red_cards = FH_data[
    (FH_data['minute'] <= 15) &
    ((FH_data['Redcards - away'] > 0) | (FH_data['Redcards - home'] > 0) | (FH_data['Yellowred Cards - home'] > 0) | (FH_data['Yellowred Cards - away'] > 0) )
]['fixture_id'].unique()
cleaned_FH_data = FH_data[~FH_data['fixture_id'].isin(matches_with_early_red_cards)]
removed_count = len(matches_with_early_red_cards)
print(f"Number of matches removed due to early red cards: {removed_count}")


#Remove matches with late goal affecting the result
SH_data = match_data[(match_data['halftime'] == '2nd-half') & (match_data['suspended'] == False) & (match_data['stopped'] == False)]
late_minute_data = SH_data[SH_data['minute'] >= 45]
matches_to_remove = []
for fixture_id in late_minute_data['fixture_id'].unique():
    match_data = SH_data[SH_data['fixture_id'] == fixture_id]
    state_at_90_plus = match_data[match_data['minute'] >= 45]['current_state'].iloc[0]
    final_result = match_data['result'].iloc[0]
    if state_at_90_plus != final_result:
        matches_to_remove.append(fixture_id)
cleaned_SH_data = SH_data[~SH_data['fixture_id'].isin(matches_to_remove)]
removed_count = len(matches_to_remove)
print(f"Number of matches removed due to late goal affecting the result: {removed_count}")


cleaned_first_half_processed = calculate_probabilities(cleaned_FH_data)
cleaned_second_half_processed = calculate_probabilities(cleaned_SH_data)
cleaned_first_half_actual = bin_and_calculate_actual(cleaned_first_half_processed, bins)
cleaned_second_half_actual = bin_and_calculate_actual(cleaned_second_half_processed, bins)
plot_draw(cleaned_first_half_processed,bin_midpoints,cleaned_first_half_actual)
plot_draw(cleaned_second_half_processed,bin_midpoints,cleaned_second_half_actual)


bins = np.arange(-1, 1.1, 0.12)
bin_midpoints = bins[:-1] + (bins[1] - bins[0]) / 2
cleaned_first_half_data_initial_bet = cleaned_FH_data.groupby('fixture_id').first()
cleaned_second_half_data_initial_bet = cleaned_SH_data.groupby('fixture_id').first()
cleaned_first_half_data_initial_bet_processed = calculate_probabilities(cleaned_first_half_data_initial_bet)
cleaned_second_half_data_initial_bet_processed = calculate_probabilities(cleaned_second_half_data_initial_bet)
cleaned_first_half_data_initial_bet_actual = bin_and_calculate_actual(cleaned_first_half_data_initial_bet_processed, bins)
cleaned_second_half_data_initial_bet_actual = bin_and_calculate_actual(cleaned_second_half_data_initial_bet_processed, bins)
plot_draw(cleaned_first_half_data_initial_bet_processed,bin_midpoints,cleaned_first_half_data_initial_bet_actual)
plot_draw(cleaned_second_half_data_initial_bet_processed,bin_midpoints,cleaned_second_half_data_initial_bet_actual)


match_data_tree =match_data[(match_data['suspended'] == False) & (match_data['stopped'] == False)]
match_data_tree.loc[:, 'result'] = match_data_tree['result'].astype('category')
match_data_tree = match_data_tree.copy()
match_data_tree.rename(columns={ '1': 'home_odds', '2': 'away_odds', 'X': 'draw_odds'}, inplace=True)
match_data_tree.columns = match_data_tree.columns.str.replace(r'[^0-9a-zA-Z_]', '_', regex=True)
r_df = pandas2ri.py2rpy(match_data_tree)

column_names = robjects.r("colnames")(r_df)
for col in column_names:
    print(col)


formula = "result ~ halftime + minute + home_odds + away_odds + draw_odds + Accurate_Crosses___away + Accurate_Crosses___home" \
          "+Assists___away + Assists___home + Attacks___away + Attacks___home + Ball_Possession_____away" \
          "+Ball_Possession_____home + Ball_Safe___away + Ball_Safe___home + Challenges___away + Challenges___home" \
          "+Corners___away + Corners___home + Counter_Attacks___away + Counter_Attacks___home + Dangerous_Attacks___away" \
          "+Dangerous_Attacks___home + Dribble_Attempts___away + Dribble_Attempts___home + Fouls___away + Fouls___home" \
          "+Free_Kicks___away +Free_Kicks___home +Goal_Attempts___away +Goal_Attempts___home +Goal_Kicks___away " \
          "+Goal_Kicks___home +Goals___away +Goals___home +Headers___away +Headers___home +Hit_Woodwork___away " \
          "+Hit_Woodwork___home +Injuries___away +Injuries___home +Interceptions___away +Interceptions___home" \
          "+Key_Passes___away +Key_Passes___home +Long_Passes___away +Long_Passes___home +Offsides___away +Offsides___home" \
          "+Passes___away +Passes___home +Penalties___away +Penalties___home +Redcards___away +Redcards___home +Saves___away " \
          "+Saves___home +Score_Change___away +Score_Change___home +Shots_Blocked___away +Shots_Blocked___home +Shots_Insidebox___away" \
          "+Shots_Insidebox___home +Shots_Off_Target___away +Shots_Off_Target___home +Shots_On_Target___away +Shots_On_Target___home" \
          "+Shots_Outsidebox___away +Shots_Outsidebox___home +Shots_Total___away +Shots_Total___home +Substitutions___away" \
          "+Substitutions___home +Successful_Dribbles___away +Successful_Dribbles___home +Successful_Headers___away" \
          "+Successful_Headers___home +Successful_Interceptions___away +Successful_Interceptions___home +Successful_Passes___away" \
          "+Successful_Passes___home +Successful_Passes_Percentage___away +Successful_Passes_Percentage___home +Tackles___away" \
          "+Tackles___home +Throwins___away +Throwins___home +Total_Crosses___away +Total_Crosses___home +Yellowcards___away " \
          "+Yellowcards___home +Yellowred_Cards___away +Yellowred_Cards___home"
control= rpart.rpart_control(cp=0.0,maxdepth=4)
r_model = r.rpart(formula, data=r_df, method="class", control= control)
print(r_model)
plot_file = "decision_tree.png"
grdevices.png(file=plot_file, width=800, height=600)
rattle.fancyRpartPlot(r_model)
grdevices.dev_off()
if os.path.exists(plot_file):
    display(Image(filename=plot_file))
else:
    print(f"Plot file {plot_file} not found!")


formula = "result ~ halftime + minute + Accurate_Crosses___away + Accurate_Crosses___home" \
          "+Assists___away + Assists___home + Attacks___away + Attacks___home + Ball_Possession_____away" \
          "+Ball_Possession_____home + Ball_Safe___away + Ball_Safe___home + Challenges___away + Challenges___home" \
          "+Corners___away + Corners___home + Counter_Attacks___away + Counter_Attacks___home + Dangerous_Attacks___away" \
          "+Dangerous_Attacks___home + Dribble_Attempts___away + Dribble_Attempts___home + Fouls___away + Fouls___home" \
          "+Free_Kicks___away +Free_Kicks___home +Goal_Attempts___away +Goal_Attempts___home +Goal_Kicks___away " \
          "+Goal_Kicks___home +Goals___away +Goals___home +Headers___away +Headers___home +Hit_Woodwork___away " \
          "+Hit_Woodwork___home +Injuries___away +Injuries___home +Interceptions___away +Interceptions___home" \
          "+Key_Passes___away +Key_Passes___home +Long_Passes___away +Long_Passes___home +Offsides___away +Offsides___home" \
          "+Passes___away +Passes___home +Penalties___away +Penalties___home +Redcards___away +Redcards___home +Saves___away " \
          "+Saves___home +Score_Change___away +Score_Change___home +Shots_Blocked___away +Shots_Blocked___home +Shots_Insidebox___away" \
          "+Shots_Insidebox___home +Shots_Off_Target___away +Shots_Off_Target___home +Shots_On_Target___away +Shots_On_Target___home" \
          "+Shots_Outsidebox___away +Shots_Outsidebox___home +Shots_Total___away +Shots_Total___home +Substitutions___away" \
          "+Substitutions___home +Successful_Dribbles___away +Successful_Dribbles___home +Successful_Headers___away" \
          "+Successful_Headers___home +Successful_Interceptions___away +Successful_Interceptions___home +Successful_Passes___away" \
          "+Successful_Passes___home +Successful_Passes_Percentage___away +Successful_Passes_Percentage___home +Tackles___away" \
          "+Tackles___home +Throwins___away +Throwins___home +Total_Crosses___away +Total_Crosses___home +Yellowcards___away " \
          "+Yellowcards___home +Yellowred_Cards___away +Yellowred_Cards___home"
control= rpart.rpart_control(cp=0.0,maxdepth=4)
r_model = r.rpart(formula, data=r_df, method="class", control= control)
print(r_model)
plot_file = "decision_tree.png"
grdevices.png(file=plot_file, width=800, height=800)
rattle.fancyRpartPlot(r_model)
grdevices.dev_off()
if os.path.exists(plot_file):
    display(Image(filename=plot_file))
else:
    print(f"Plot file {plot_file} not found!")

match_data_tree_first_half =cleaned_FH_data
match_data_tree_first_half.loc[:, 'result'] = match_data_tree_first_half['result'].astype('category')
match_data_tree_first_half = match_data_tree_first_half.copy()
match_data_tree_first_half.rename(columns={ '1': 'home_odds', '2': 'away_odds', 'X': 'draw_odds'}, inplace=True)
match_data_tree_first_half.columns = match_data_tree_first_half.columns.str.replace(r'[^0-9a-zA-Z_]', '_', regex=True)
r_df = pandas2ri.py2rpy(match_data_tree_first_half)
formula = "result ~ minute + home_odds + away_odds + draw_odds + Accurate_Crosses___away + Accurate_Crosses___home" \
          "+Assists___away + Assists___home + Attacks___away + Attacks___home + Ball_Possession_____away" \
          "+Ball_Possession_____home + Ball_Safe___away + Ball_Safe___home + Challenges___away + Challenges___home" \
          "+Corners___away + Corners___home + Counter_Attacks___away + Counter_Attacks___home + Dangerous_Attacks___away" \
          "+Dangerous_Attacks___home + Dribble_Attempts___away + Dribble_Attempts___home + Fouls___away + Fouls___home" \
          "+Free_Kicks___away +Free_Kicks___home +Goal_Attempts___away +Goal_Attempts___home +Goal_Kicks___away " \
          "+Goal_Kicks___home +Goals___away +Goals___home +Headers___away +Headers___home +Hit_Woodwork___away " \
          "+Hit_Woodwork___home +Injuries___away +Injuries___home +Interceptions___away +Interceptions___home" \
          "+Key_Passes___away +Key_Passes___home +Long_Passes___away +Long_Passes___home +Offsides___away +Offsides___home" \
          "+Passes___away +Passes___home +Penalties___away +Penalties___home +Redcards___away +Redcards___home +Saves___away " \
          "+Saves___home +Score_Change___away +Score_Change___home +Shots_Blocked___away +Shots_Blocked___home +Shots_Insidebox___away" \
          "+Shots_Insidebox___home +Shots_Off_Target___away +Shots_Off_Target___home +Shots_On_Target___away +Shots_On_Target___home" \
          "+Shots_Outsidebox___away +Shots_Outsidebox___home +Shots_Total___away +Shots_Total___home +Substitutions___away" \
          "+Substitutions___home +Successful_Dribbles___away +Successful_Dribbles___home +Successful_Headers___away" \
          "+Successful_Headers___home +Successful_Interceptions___away +Successful_Interceptions___home +Successful_Passes___away" \
          "+Successful_Passes___home +Successful_Passes_Percentage___away +Successful_Passes_Percentage___home +Tackles___away" \
          "+Tackles___home +Throwins___away +Throwins___home +Total_Crosses___away +Total_Crosses___home +Yellowcards___away " \
          "+Yellowcards___home +Yellowred_Cards___away +Yellowred_Cards___home"
control= rpart.rpart_control(cp=0.0,maxdepth=4)
r_model = r.rpart(formula, data=r_df, method="class", control= control)
print(r_model)
plot_file = "decision_tree.png"
grdevices.png(file=plot_file, width=800, height=800)
rattle.fancyRpartPlot(r_model)
grdevices.dev_off()
if os.path.exists(plot_file):
    display(Image(filename=plot_file))
else:
    print(f"Plot file {plot_file} not found!")

match_data_tree_first_half =cleaned_SH_data
match_data_tree_first_half.loc[:, 'result'] = match_data_tree_first_half['result'].astype('category')
match_data_tree_first_half = match_data_tree_first_half.copy()
match_data_tree_first_half.rename(columns={ '1': 'home_odds', '2': 'away_odds', 'X': 'draw_odds'}, inplace=True)
match_data_tree_first_half.columns = match_data_tree_first_half.columns.str.replace(r'[^0-9a-zA-Z_]', '_', regex=True)
r_df = pandas2ri.py2rpy(match_data_tree_first_half)
formula = "result ~ minute + home_odds + away_odds + draw_odds + Accurate_Crosses___away + Accurate_Crosses___home" \
          "+Assists___away + Assists___home + Attacks___away + Attacks___home + Ball_Possession_____away" \
          "+Ball_Possession_____home + Ball_Safe___away + Ball_Safe___home + Challenges___away + Challenges___home" \
          "+Corners___away + Corners___home + Counter_Attacks___away + Counter_Attacks___home + Dangerous_Attacks___away" \
          "+Dangerous_Attacks___home + Dribble_Attempts___away + Dribble_Attempts___home + Fouls___away + Fouls___home" \
          "+Free_Kicks___away +Free_Kicks___home +Goal_Attempts___away +Goal_Attempts___home +Goal_Kicks___away " \
          "+Goal_Kicks___home +Goals___away +Goals___home +Headers___away +Headers___home +Hit_Woodwork___away " \
          "+Hit_Woodwork___home +Injuries___away +Injuries___home +Interceptions___away +Interceptions___home" \
          "+Key_Passes___away +Key_Passes___home +Long_Passes___away +Long_Passes___home +Offsides___away +Offsides___home" \
          "+Passes___away +Passes___home +Penalties___away +Penalties___home +Redcards___away +Redcards___home +Saves___away " \
          "+Saves___home +Score_Change___away +Score_Change___home +Shots_Blocked___away +Shots_Blocked___home +Shots_Insidebox___away" \
          "+Shots_Insidebox___home +Shots_Off_Target___away +Shots_Off_Target___home +Shots_On_Target___away +Shots_On_Target___home" \
          "+Shots_Outsidebox___away +Shots_Outsidebox___home +Shots_Total___away +Shots_Total___home +Substitutions___away" \
          "+Substitutions___home +Successful_Dribbles___away +Successful_Dribbles___home +Successful_Headers___away" \
          "+Successful_Headers___home +Successful_Interceptions___away +Successful_Interceptions___home +Successful_Passes___away" \
          "+Successful_Passes___home +Successful_Passes_Percentage___away +Successful_Passes_Percentage___home +Tackles___away" \
          "+Tackles___home +Throwins___away +Throwins___home +Total_Crosses___away +Total_Crosses___home +Yellowcards___away " \
          "+Yellowcards___home +Yellowred_Cards___away +Yellowred_Cards___home"
control= rpart.rpart_control(cp=0.0,maxdepth=4)
r_model = r.rpart(formula, data=r_df, method="class", control= control)
print(r_model)
plot_file = "decision_tree.png"
grdevices.png(file=plot_file, width=800, height=800)
rattle.fancyRpartPlot(r_model)
grdevices.dev_off()
if os.path.exists(plot_file):
    display(Image(filename=plot_file))
else:
    print(f"Plot file {plot_file} not found!")

control= rpart.rpart_control(cp=0.0,maxdepth=3)
r_model = r.rpart(formula, data=r_df, method="class", control= control)
print(r_model)
plot_file = "decision_tree.png"
grdevices.png(file=plot_file, width=800, height=800)
rattle.fancyRpartPlot(r_model)
grdevices.dev_off()
if os.path.exists(plot_file):
    display(Image(filename=plot_file))
else:
    print(f"Plot file {plot_file} not found!")