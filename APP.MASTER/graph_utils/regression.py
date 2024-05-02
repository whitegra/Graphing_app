import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def plot_linear_regression(df, x_col, y_col, save_path):
    plt.figure(figsize=(10, 8))  # Increase size of the graph
    
    x = df[x_col].values.reshape(-1, 1)
    y = df[y_col].values
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    
    # Plot scatter plot and regression line
    plt.scatter(x, y, label="Data points")
    plt.plot(x, y_pred, color='red', label=f'Regression line: y = {model.intercept_:.2f} + {model.coef_[0]:.2f}')
    
    # Calculate R squared
    r_squared = r2_score(y, y_pred)
    
    # Add R squared and regression line equation to legend
    plt.legend(loc='upper right', fontsize=10)
    plt.legend([f'Data points', f'Regression line: y = {model.intercept_:.2f} + {model.coef_[0]:.2f} \nR squared: {r_squared:.2f}'], loc='upper right', fontsize=10)
    
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('Linear Regression Plot')
    plt.savefig(save_path)
    plt.close()
