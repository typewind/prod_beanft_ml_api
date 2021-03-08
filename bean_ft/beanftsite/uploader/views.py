from flask import render_template, abort, url_for, flash, redirect, request, Blueprint, Response, session, jsonify, make_response
from flask_bootstrap import Bootstrap
from flask_login import current_user, login_required
from beanftsite import db
from beanftsite.models import FileContents, ModelType, ListXY, factorise_data, convert_df_integer_to_numeric, convert_array_integer_to_numeric
from werkzeug.utils import secure_filename
import os
import urllib.request
import pandas as pd
import numpy as np
import io
import base64
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sqlalchemy import event, Column, Integer, String, create_engine

# ML Packages 
## Categorical
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

## test api
import requests
import json
import pickle
import pylint
import warnings
warnings.filterwarnings('ignore')

## Regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import linear_model
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder



# Configuration settings
uploader = Blueprint('uploader', __name__)
basedir = os.path.abspath(os.path.dirname(__name__))
folder = os.path.abspath(basedir + str('\\beanftsite\\static\\data_uploads\\'))
ALLOWED_EXTENSIONS = set(['csv'])


# Core code
@uploader.route('/uploads', methods=['GET', 'POST'])
def uploads():
    return render_template('uploads.html')

@uploader.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST' and 'inputFiles' in request.files:
        file = request.files['inputFiles']
        filename = secure_filename(file.filename)
        data_reload = FileContents(name=filename)
        db.create_all()
        db.session.add(data_reload)
        db.session.commit()
        data_reloaded = FileContents.query.all()
        # os.path.join is used so that paths work in every operating system
        file.save(os.path.join(folder,filename))
        # Data review
        new_data = pd.read_csv(os.path.join(folder,str(data_reloaded[-1])))
        new_dataplot = new_data.head(10)
        new_data_info = new_data.info
        new_data_size = new_data.size
        new_data_shape = new_data.shape
        dropdown_list = list(new_data.columns)
        flash('Just a moment, app is thinking!')

        if str(data_reloaded[-1]).split('.')[-1] != 'csv':
            # Forbidden, No Access
            abort(403)
    return render_template('train.html',
        new_dataplot = new_dataplot,
        data_reload = data_reload,
        new_data_info = new_data.info,
        new_data_size=new_data_size,
        new_data_shape=new_data_shape,
        dropdown_list=dropdown_list
        )


@uploader.route('/fit', methods=['GET', 'POST'])
def fit():
    # X and Y vars
    y_var_select = request.form.get('y_var')
    multiselect = request.form.getlist('x_vars')

    # Type of forecast
    pred_type_select = request.form.get('rd_pred_type')
    
    # commit the prediction type
    pred_type_selected = ModelType(pred_type_select)
    db.create_all()
    db.session.add(pred_type_selected)

    # commit the X and Y vars
    xy_selection = ListXY(y_var_select,multiselect)
    db.create_all()
    db.session.add(xy_selection)
    db.session.commit()

    # testing - DELETE AFTEWARDS
    ListXY.query.all()

    # select vars
    data_reloaded = FileContents.query.all()
    new_data = pd.read_csv(os.path.join(folder, str(data_reloaded[-1])))
    new_data = new_data.dropna() # deletes Na and NaN
    X = new_data[multiselect]
    Y = new_data[y_var_select]

    if request.form.get('rd_pred_type') == "Classification":
        # Step 1: Refactor columns with text to integer and remove NAs
        X = factorise_data(X)

        # prepare models
        seed = 7
        models = []
        models.append(('RandomForestClassifier', RandomForestClassifier()))
        models.append(('GradientBoostingClassifier', GradientBoostingClassifier()))
        models.append(('LogisticRegression', LogisticRegression()))
        models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
        models.append(('KNeighborsClassifier', KNeighborsClassifier()))
        models.append(('GaussianNB', GaussianNB()))
        models.append(('SVC', SVC()))
        # evaluate each model in turn
        results = []
        names = []
        allmodels = []
        scoring = 'accuracy'
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s - %f | %f" % (name, cv_results.mean(), cv_results.std())
            allmodels.append(msg)
            model_results = results
            model_names = names


    if request.form.get('rd_pred_type') == "Regression":
        # Step 1: Refactor columns with text to integer and remove NAs
        X = factorise_data(X)

        # prepare models
        models = []
        models.append(('RandomForestRegressor', RandomForestRegressor(n_estimators=200)))
        models.append(('GradientBoostingRegressor', GradientBoostingRegressor(n_estimators=200)))
        models.append(('Ridge', Ridge()))
        models.append(('ElasticNet', ElasticNet()))
        models.append(('Lasso', Lasso()))
        models.append(('SVR', SVR()))
        # evaluate each model in turn
        results = []
        names = []
        allmodels = []
        for name, model in models:
            X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 7)
            # standard scaler #standardises the feature variables
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            model_to_fit = model
            model_to_fit.fit(X_train, y_train)
            predictions = model_to_fit.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            results.append(mse)
            names.append(name)
            msg = "%s - %.2f | %s" % (name, (np.sqrt(mse)), "-")
            allmodels.append(msg)
            model_results = results
            model_names = names

    return render_template('fit.html',y_var_select=y_var_select,
        pred_type_select = pred_type_select,
        multiselect = multiselect, 
        model_results = allmodels,
        model_names = names)


@uploader.route("/predict_data" , methods=['GET', 'POST'])
def predict_data():

    if request.method == 'POST' and 'inputTestFile' in request.files:
        # load selected model type
        selected_model_type = request.form.get('selected_model')
        # load the data to predict on
        testfile = request.files['inputTestFile']
        testfilename = secure_filename(testfile.filename)
        testfile.save(os.path.join(folder,testfilename))

        # load the previously selected prediction type, X and Y vars
        pred_type_select_all = ModelType.query.all()
        x_selected_model = ListXY.query.order_by(ListXY.id.desc()).first().x_vars
        y_selected_model = ListXY.query.order_by(ListXY.id.desc()).first().y_var
        pred_type = str(pred_type_select_all[-1])

        
        # load data again to predict
        data_reloaded = FileContents.query.all()
        data_reloaded_2 = str(data_reloaded[-1])
        new_data = pd.read_csv(os.path.join(folder,str(data_reloaded[-1])))
        new_data = new_data.dropna() # deletes Na and NaN

        X = new_data[eval(x_selected_model)]
        Y = new_data[y_selected_model]

        if pred_type == "Classification":
            # Step 1: Refactor columns with text to integer and remove NAs
            X = factorise_data(X)

            # prepare models
            seed = 7
            models = []
            models.append(('RandomForestClassifier', RandomForestClassifier(n_estimators=200)))
            models.append(('GradientBoostingClassifier', GradientBoostingClassifier(n_estimators=200)))
            models.append(('LogisticRegression', LogisticRegression()))
            models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
            models.append(('KNeighborsClassifier', KNeighborsClassifier()))
            models.append(('GaussianNB', GaussianNB()))
            models.append(('SVC', SVC()))

            # evaluate each model in turn
            results = []
            names = []
            allmodels = []
            scoring = 'accuracy'
            for name, model in models:
                X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 7)
                # standard scaler #standardises the feature variables
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)
                if selected_model_type == name:
                    model_to_fit = model
                    model_to_fit.fit(X_train, Y_train)
                    # save model and test api
                    saved_model = os.path.join(folder, str("final_model.sav")) 
                    pickle.dump(model_to_fit,open(saved_model, 'wb'))
                    # load data to predict on
                    testfilename_csv = pd.read_csv(folder + str("\\") + str(testfilename), dtype=str)
                    test_data = testfilename_csv.dropna() # deletes Na and NaN
                    # Using Classes, Refactor columns with text to integer and remove NAs
                    test_data = factorise_data(test_data)
                    # Predict on the loaded data, first scale it
                    data = sc.fit_transform(test_data)
                    predictions = model_to_fit.predict(data)
                    # Use classes to apply the int to float function
                    predictions = convert_array_integer_to_numeric(predictions)

        if pred_type == "Regression":
            # Step 1: Refactor columns with text to integer and remove NAs
            X = factorise_data(X)

            # prepare models
            models = []
            models.append(('RandomForestRegressor', RandomForestRegressor(n_estimators=200)))
            models.append(('GradientBoostingRegressor', GradientBoostingRegressor(n_estimators=200)))
            models.append(('Ridge', Ridge()))
            models.append(('ElasticNet', ElasticNet()))
            models.append(('Lasso', Lasso()))
            models.append(('SVR', SVR()))

            # evaluate each model in turn
            results = []
            names = []
            allmodels = []
            for name, model in models:
                X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 7)
                # standard scaler #standardises the feature variables
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)
                if selected_model_type == name:
                    model_to_fit = model
                    model_to_fit.fit(X_train, y_train)
                    # save model and test api
                    saved_model = os.path.join(folder, str("final_model.sav")) 
                    pickle.dump(model_to_fit,open(saved_model, 'wb'))
                    # load data to predict on
                    testfilename_csv = pd.read_csv(folder + str("\\") + str(testfilename), dtype=str)
                    test_data = testfilename_csv.dropna() # deletes Na and NaN
                    # Step 1: Using Classes, Refactor columns with text to integer and remove NAs
                    test_data = factorise_data(test_data)
                    # Predict on the loaded data, first scale it
                    data = sc.transform(test_data)
                    predictions = model_to_fit.predict(data)
                    # Use classes to apply the int to float function
                    predictions = convert_array_integer_to_numeric(predictions)
    else:
        return jsonify("Error occured while preprocessing your data for our model!")
      
    print("Your model has been saved") 
    # predictions.to_csv(os.path.join(folder,str(testfilename),str("_predicted.csv"))  
    response={'data':[]}
    # response={'data':[],'prediction_label':{'species':"setosa",'species':"versicolor",'species':"virginica"}}
    response['data']=list(predictions)  
    return make_response(jsonify(response),200)



# PLOT ----------------------------------------------------------

@uploader.route('/plot', methods=['GET', 'POST'])
def plot():
    return render_template('plot.html')

@uploader.route("/plot_2" , methods=['GET', 'POST'])
def plot_2():
    if request.method == 'POST' and 'inputFiles' in request.files:
        file = request.files['inputFiles']
        filename = secure_filename(file.filename)
        data_reload = FileContents(name=filename)
        db.create_all()
        db.session.add(data_reload)
        db.session.commit()
        data_reload = FileContents.query.all()
        file.save(os.path.join(folder,filename))
        new_data = pd.read_csv(os.path.join(folder, str(data_reload[-1])))
        dropdown_list = list(new_data.columns)
        return render_template('plot_2.html',
            dropdown_list = dropdown_list)
        

@uploader.route('/graph', methods=['GET', 'POST'])
def chart():
    x_axis_select = request.form.get('select_x')
    y_axis_select = request.form.get('select_y')
    x_axis_select_str = str(x_axis_select) 
    y_axis_select_str = str(y_axis_select) 

    data_reload = FileContents.query.all()
    new_data = pd.read_csv(os.path.join(folder,str(data_reload[-1])))
    new_data = new_data.dropna()
    x = new_data[x_axis_select_str]
    y = new_data[y_axis_select_str]
    legend = 'Monthly Data'
    labels = sorted(x)
    values = sorted(y)
    return render_template('chart.html', values=values, labels=labels, legend=legend,
    x_axis_select_str = x_axis_select_str, y_axis_select_str = y_axis_select_str)