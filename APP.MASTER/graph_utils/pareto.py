import numpy as np
import matplotlib.pyplot as plt

def plot_pareto(df, x_col, y_col, save_path):
    def identify_pareto(scores):
        is_dominated = np.zeros(scores.shape[0], dtype=bool)
        for i, c in enumerate(scores):
            if not is_dominated[i]:
                is_dominated |= (scores <= c).all(axis=1) & (scores < c).any(axis=1)
                is_dominated[i] = False
        return ~is_dominated
    
    scores = np.column_stack((df[x_col].values, df[y_col].values))
    pareto_indices = identify_pareto(scores)
    pareto_front_df = df.iloc[pareto_indices]
    pareto_front_df = pareto_front_df.sort_values(by=[x_col, y_col])

    plt.figure(figsize=(10, 8))
    plt.scatter(df[x_col], df[y_col], alpha=0.4, label='Design Points')
    plt.plot(pareto_front_df[x_col], pareto_front_df[y_col], color='red', label='Pareto Frontier')
    plt.scatter(pareto_front_df[x_col], pareto_front_df[y_col], color='red', zorder=5)
    plt.title('Pareto Frontier')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=900)
    plt.show()


