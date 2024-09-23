import matplotlib.pyplot as plt

def plot_leaderboard(leaderboard_df, figsize=(10, 5), orient='v'):
    """
    Plot the leaderboard

    Parameters
    ----------
    leaderboard_df : pd.DataFrame
    """
    plt.figure(figsize=figsize)
    if orient == 'h':
        plt.barh(leaderboard_df['model'], abs(leaderboard_df['score_val']))
    elif orient == 'v':
        plt.plot(leaderboard_df['model'], abs(leaderboard_df['score_val']), marker='o')
        plt.xticks(rotation=90)  # Rotate x-axis labels 90 degrees
    plt.title('Validation Scores')
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.grid()
    plt.show()