from flask import Flask, render_template, request
import os
import pandas as pd
from graph_utils.line import plot_line
from graph_utils.scatter import plot_scatter
from graph_utils.regression import plot_linear_regression
from graph_utils.pareto import plot_pareto

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
app.config['SECRET_KEY'] = "potato"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if not file or not allowed_file(file.filename):
                return render_template('index.html', image=None, error="Invalid file type.")
            filename = 'uploaded.csv'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            df = pd.read_csv(filepath)
            x_column = request.form.get('x_column')
            y_column = request.form.get('y_column')
            if not x_column or not y_column:
                return render_template('index.html', image=None, error="Please provide both X and Y column names.")
        else:
            x_values = request.form.get('x_values')
            y_values = request.form.get('y_values')
            if not x_values or not y_values:
                return render_template('index.html', image=None, error="Please provide both X and Y values.")
            df = pd.DataFrame({'x': x_values.split(), 'y': y_values.split()})
            x_column = 'x'
            y_column = 'y'

        graph_type = request.form['graph_type']
        output_filename = 'output.png'
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        if graph_type == 'line':
            plot_line(df, x_column, y_column, output_filepath, request.form.get('x_values'), request.form.get('y_values'))
        elif graph_type == 'scatter':
            plot_scatter(df, x_column, y_column, output_filepath)
        elif graph_type == 'linear_regression':
            plot_linear_regression(df, x_column, y_column, output_filepath)
        elif graph_type == 'pareto':
            plot_pareto(df, x_column, y_column, output_filepath)
        else:
            return render_template('index.html', image=None, error="Invalid graph type.")

        return render_template(f'index/index_{graph_type}.html', image=output_filename)

    return render_template('index.html', image=None)

@app.route('/index/index_line.html')
def index_line():
    return render_template('index/index_line.html', image=None)

@app.route('/index/index_scatter.html')
def index_scatter():
    return render_template('index/index_scatter.html', image=None)

@app.route('/index/index_linear_regression.html')
def index_linear_regression():
    return render_template('index/index_linear_regression.html', image=None)

@app.route('/index/index_pareto.html')
def index_pareto():
    return render_template('index/index_pareto.html', image=None)

if __name__ == '__main__':
    app.run(debug=True)

