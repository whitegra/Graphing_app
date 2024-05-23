import matplotlib.pyplot as plt

def plot_scatter(df, x_col, y_col, save_path):
    plt.figure(figsize=(10, 8)) 
    # Plotting x values in red against their index
    plt.scatter(df.index, df[x_col], color='red', label=f'{x_col} values')
    # Plotting y values in blue against their index
    plt.scatter(df.index, df[y_col], color='blue', label=f'{y_col} values')

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('Scatter Plot')
    plt.legend()  # Add a legend to clarify the colors
    plt.savefig(save_path, dpi=900)
    plt.close()
