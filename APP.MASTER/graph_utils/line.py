import matplotlib.pyplot as plt

def plot_line(df, x_col, y_col, save_path):
    plt.figure()
    plt.plot(df[x_col], df[y_col], marker='o')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('Line Graph')
    plt.savefig(save_path)
    plt.close()


