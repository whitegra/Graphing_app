from flask import Flask, render_template, request
import os
from werkzeug.exceptions import NotFound as TemplateNotFound 
import pandas as pd
from graph_utils.line import plot_line
from graph_utils.scatter import plot_scatter
from graph_utils.regression import plot_linear_regression
from graph_utils.pareto import plot_pareto

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
app.config['SECRET_KEY'] = "potato"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if not file or not allowed_file(file.filename):
            return render_template('index.html', image=None, error="Invalid file type.")
        x_column = request.form['x_column']
        y_column = request.form['y_column']
        x2_column = request.form['x2_column']
        y2_column = request.form['y2_column']
        graph_type = request.form['graph_type']
        
        filename = 'uploaded.csv'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        df = pd.read_csv(filepath)
        
        output_filename = 'output.png'
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        if graph_type == 'line':
            plot_line(df, x_column, y_column, output_filepath)
        elif graph_type == 'scatter':
            plot_scatter(df, x_column, y_column, output_filepath)
        elif graph_type == 'linear_regression':
            plot_linear_regression(df, x_column, y_column, output_filepath)
        elif graph_type == 'pareto_front':
            plot_pareto(df, x_column, y_column, output_filepath)
        else:
            return render_template('index.html', image=None, error="Invalid graph type.")
        
        return render_template('index.html', image=output_filename)
    
    return render_template('index.html', image=None)

@app.route('/help/<graph_type>')
def help_content(graph_type):
    try:
        return render_template(f'help_content/help_{graph_type}.html')
    except TemplateNotFound:
        return "Help content not found", 404

if __name__ == '__main__':
    app.run(debug=True)
