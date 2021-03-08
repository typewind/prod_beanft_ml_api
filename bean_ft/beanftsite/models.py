from beanftsite import db,login_manager
from datetime import datetime
from werkzeug.security import generate_password_hash,check_password_hash
from flask_login import UserMixin
import numpy as np
import pandas as pd 

# By inheriting the UserMixin we get access to a lot of built-in attributes
# which we will be able to call in our views!
# is_authenticated()
# is_active()
# is_anonymous()
# get_id()


# The user_loader decorator allows flask-login to load the current user
# and grab their id.

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)

class User(db.Model, UserMixin):

    # Create a table in the db
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key = True)
    profile_image = db.Column(db.String(20), nullable=False, default='default_profile.png')
    email = db.Column(db.String(64), unique=True, index=True)
    username = db.Column(db.String(64), unique=True, index=True)
    password_hash = db.Column(db.String(128))
    # This connects BlogPosts to a User Author.
    
    def __init__(self, email, username, password):
        self.email = email
        self.username = username
        self.password_hash = generate_password_hash(password)

    def check_password(self,password):
        # https://stackoverflow.com/questions/23432478/flask-generate-password-hash-not-constant-output
        return check_password_hash(self.password_hash,password)

    def __repr__(self):
        return f"UserName: {self.username}"

class FileContents(db.Model):
    __tablename__ = 'mydata'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300))
    # data = db.Column(db.LargeBinary)

    def __init__(self, name):
        self.name = name
        # self.data = data

    def __repr__(self):
        return f"{self.name}"

class ModelType(db.Model):
    __tablename__ = 'mymodel'

    id = db.Column(db.Integer, primary_key=True)
    md_type = db.Column(db.String(300))

    def __init__(self, md_type):
        self.md_type = md_type

    def __repr__(self):
        return f"{self.md_type}"


class ListXY(db.Model):
    __tablename__ = 'my_xy'

    id = db.Column(db.Integer, primary_key=True)
    y_var = db.Column(db.String(300))
    x_vars = db.Column(db.String(300))

    def __init__(self,y_var,x_vars):
        self.y_var = y_var
        self.x_vars = str(x_vars)

    def __repr__(self):
        return f"{self.y_var,self.x_vars}"

# class LoadDataframe():
#     def __init__(self,df):
#         self.df = df
#     def factorise_data(self,df):
#         for i in df:
#             if df.dtypes[i] != np.float64 or np.int64:
#                 df[i], _ = pd.factorize(df[i],sort = True)
#         print(df)
            
#     def convert_integer_to_numeric(self,df):
#         for i in df:
#             if df.dtypes[i] == np.int64:
#                 df[i] = df[i].astype(np.float64)
#                 df.round(2)
#         print(df)

# Preprocessing functions

def factorise_data(df):
    for i in df:
        if df.dtypes[i] != np.float64 or np.int64:
            df[i], _ = pd.factorize(df[i],sort = True)
    return(df)
        
def convert_df_integer_to_numeric(df):
    for i in df:
        if df.dtypes[i] == np.int64:
            df[i] = df[i].astype(np.float64)
            df.round(2)
    return(df)

def convert_array_integer_to_numeric(array_obj):
    if array_obj.dtype == 'int64':
        array_obj = array_obj.astype(np.float64)
    return(array_obj)
