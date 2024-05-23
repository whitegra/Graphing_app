import matplotlib.pyplot as plt

def plot_line(df, x_column, y_column, save_path, x_values=None, y_values=None):
    if x_values is not None and y_values is not None:
        x_vals = [float(val) for val in x_values.split()]
        y_vals = [float(val) for val in y_values.split()]
    else:
        x_vals = df[x_column]
        y_vals = df[y_column]

    plt.figure(figsize=(10, 8))
    plt.plot(x_vals, y_vals, marker='o')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Line Graph')
    plt.savefig(save_path, dpi=900)
    plt.close()
