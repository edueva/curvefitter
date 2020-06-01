import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def read_file(file_path):
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, delimiter=";")
        if df.shape[1] == 1:
            df = pd.read_excel(file_path, delimiter=";", decimal=",")
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        if df.shape[1] == 1:
            df = pd.read_csv(file_path, delimiter=";")
    return df


def pl4(x, A, B, C, D):
    """4PL lgoistic equation."""
    return ((A - D) / (1.0 + ((x / C) ** B))) + D


def pl5(x, A, B, C, D, E):
    """5PL lgoistic equation."""
    return ((A - D) / (1.0 + ((x / C) ** B) ** E)) + D


functions = {'4PL': pl4,
             '5PL': pl5}


def fit(fun, x, y):
    popt, pcov = curve_fit(fun, x, y)
    return popt, pcov


def plot(dataframe, fun, x, y, err=''):
    popt, pcov = fit(fun, dataframe[x], dataframe[y])
    df['fitted_y'] = fun(dataframe[x], *popt)
    # R_squared = round(1 - (np.sum(np.square(dataframe[y] - df['fitted_y'])))/\
    #            (np.sum(np.square(dataframe[y] - dataframe[y].mean()))), 3)
    if fun == pl4:
        label_text = '{} A=%5.3f, B=%5.3f, C=%5.3f, D=%5.3f'.format('fit 4PL')
    elif fun == pl5:
        label_text = '{} A=%5.3f, B=%5.3f, C=%5.3f, D=%5.3f, E=%5.3f'.format('fit 5PL')
    plt.plot(dataframe[x], dataframe['fitted_y'], label=label_text % tuple(popt))

    try:
        plt.errorbar(dataframe[x], dataframe[y], dataframe[err], fmt='o', label='data {}'.format(y))
    except KeyError:
        plt.plot(dataframe[x], dataframe[y], 'o', label='data {}'.format(y))


def log_transform(dataframe, col_name, err_col_name=''):
    log_col_name = 'log10({})'.format(col_name)
    dataframe[log_col_name] = np.log10(dataframe[col_name])
    dataframe[log_col_name].replace([np.inf, -np.inf], 0.0, inplace=True)
    rel_std_col_name = 'rel_std({})'.format(str(log_col_name))
    if err_col_name != '':
        dataframe[rel_std_col_name] = (1 / np.log(10)) * (dataframe[err_col_name] / dataframe[col_name])
    return dataframe[log_col_name], log_col_name, rel_std_col_name


def find_mean(dataframe, col_names):
    new_col_name = '; '.join([str(elem) for elem in col_names])
    mean_col_name = 'mean({})'.format(new_col_name)
    std_col_name = 'std({})'.format(str(new_col_name))
    dataframe[mean_col_name] = dataframe[col_names].mean(axis=1)
    dataframe[std_col_name] = dataframe[col_names].std(axis=1)
    return dataframe[mean_col_name], mean_col_name, std_col_name


def work_Y(seq_num):
    Y = st.sidebar.multiselect('Select Y(s) ({})'.format(seq_num), initial_cols)
    log_Y = []
    mean_Y = []

    handles_Y = [Y, log_Y, mean_Y]

    if len(Y) > 1:
        Y_handle = 2
        col, new_Y, std = find_mean(df, Y)
        mean_Y.extend([new_Y, std])
    else:
        Y_handle = 0

    sb_log_Y = st.sidebar.checkbox("Transform Y(s) ({}) to Log10?".format(seq_num))

    if sb_log_Y:
        try:
            col, new_Y, std = log_transform(df, *handles_Y[Y_handle])
            log_Y.extend([new_Y, std])
            Y_handle = 1
        except TypeError:
            st.write('<-- Select Y(s)')

    sb_func = st.sidebar.selectbox('Select equation type ({})'.format(seq_num), ['4PL', "5PL"])
    try:
        plot(df, functions[sb_func], final_X, *handles_Y[Y_handle])
    except (NameError, TypeError):
        pass
    except RuntimeError:
        st.write('Optimal parameters not found')


'''
# CurveFitter
This very simple app allows you to fit 4- and 5-parametric curves
'''

#  /Users/dueva1/Documents/Projects/NLRP3/IL_controls.csv

filename = st.text_input('Enter a file path (it should look like /Users/dueva/Documents/Projects/NLRP3/test.xlsx)')
if filename != '':
    df = read_file(filename)
    st.write(df)
    initial_cols = list(df)

    w = 10
    h = 7
    d = 70
    plt.figure(figsize=(w, h))

    X = st.sidebar.multiselect('Select X', initial_cols)
    X_handle = 0
    try:
        logging_X = st.sidebar.checkbox('Transform X to Log10?')
        handles_X = [X]

        if logging_X:
            X_handle = 1
            col, new_X, std_X = log_transform(df, X[0])
            handles_X.append([new_X])
        final_X = handles_X[X_handle][0]
    except (IndexError):
        st.write('<--- Select X')

    seq_num_fl = 1
    work_Y(seq_num_fl)
    seq_num_fl += 1

    while st.sidebar.checkbox('Add {} curve?'.format(seq_num_fl)):
        work_Y(seq_num_fl)
        seq_num_fl += 1

    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.70])
    legend_x = 0.5
    legend_y = 1.3
    plt.legend(loc='upper center', bbox_to_anchor=(legend_x, legend_y))
    try:
        plt.xlabel(final_X)
    except NameError:
        pass
    plt.ylabel('Y')
    st.pyplot()