pip install matplotlib
import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from graph_utils.line import plot_line
from graph_utils.scatter import plot_scatter
from graph_utils.regression import plot_linear_regression
from graph_utils.pareto import plot_pareto

st.set_page_config(page_title='Graph Generator', page_icon=':chart_with_upwards_trend:')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}

def upload_file():
    st.title('Upload CSV File for Graph Generation')
    file = st.file_uploader("Choose a CSV file", type="csv")
    if file is not None:
        if not allowed_file(file.name):
            st.error("Invalid file type.")
            return
        x_column = st.text_input("Enter X Column Name")
        y_column = st.text_input("Enter Y Column Name")
        x2_column = st.text_input("Enter X2 Column Name")
        y2_column = st.text_input("Enter Y2 Column Name")
        graph_type = st.selectbox(
            "Select Graph Type",
            ("line", "scatter", "linear_regression", "pareto_front")
        )
        if st.button("Generate Graph"):
            filepath = os.path.join('static/images', 'uploaded.csv')
            with open(filepath, 'wb') as f:
                f.write(file.getvalue())
            df = pd.read_csv(filepath)
            output_filename = 'output.png'
            output_filepath = os.path.join('static/images', output_filename)
            if graph_type == 'line':
                plot_line(df, x_column, y_column, output_filepath)
            elif graph_type == 'scatter':
                plot_scatter(df, x_column, y_column, x2_column, y2_column, output_filepath)
            elif graph_type == 'linear_regression':
                plot_linear_regression(df, x_column, y_column, output_filepath)
            elif graph_type == 'pareto_front':
                plot_pareto(df, x_column, y_column, output_filepath)
            else:
                st.error("Invalid graph type.")
                return
            st.image(output_filepath, use_column_width=True)

def help_content(graph_type):
    help_text = {
        'line': "Help content for line graph.",
        'scatter': "Help content for scatter plot.",
        'linear_regression': "Help content for linear regression.",
        'pareto_front': "Help content for pareto front graph."
    }
    return help_text.get(graph_type, "Help content not found")

def main():
    upload_file()
    graph_type = st.selectbox(
        "Select Graph Type to Get Help",
        ("line", "scatter", "linear_regression", "pareto_front")
    )
    if st.button("Get Help"):
        st.write(help_content(graph_type))

if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-

