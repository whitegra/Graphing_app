import matplotlib.pyplot as plt

def plot_scatter(df, x_col, y_col, save_path):
    plt.figure()
    # Plotting x values in red against their index
    plt.scatter(df.index, df[x_col], color='red', label=f'{x_col} values')
    # Plotting y values in blue against their index
    plt.scatter(df.index, df[y_col], color='blue', label=f'{y_col} values')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Graph of X and Y Values')
    plt.legend()  # Add a legend to clarify the colors
    plt.savefig(save_path)
    plt.close()
