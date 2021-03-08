import os
from flask import Flask
# from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager



app = Flask(__name__)
# Bootstrap(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# app = Flask(__name__,template_folder='./templates',static_folder='./static')

#############################################################################
############ CONFIGURATIONS (CAN BE SEPARATE CONFIG.PY FILE) ###############
###########################################################################

# Remember you need to set your environment variables at the command line
# when you deploy this to a real website.
# export SECRET_KEY=mysecret
# set SECRET_KEY=mysecret
app.config['SECRET_KEY'] = 'mysecret'

#################################
### DATABASE SETUPS ############
###############################

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config["CACHE_TYPE"] = "null"

ALLOWED_EXTENSIONS = set(['txt', 'csv', 'xlsx', 'jpg', 'jpeg', 'gif'])

db = SQLAlchemy(app)
Migrate(app,db)


###########################
#### LOGIN CONFIGS #######
#########################

login_manager = LoginManager()

# We can now pass in our app to the login manager
login_manager.init_app(app)

# Tell users what view to go to when they need to login.
login_manager.login_view = "users.login"


###########################
#### BLUEPRINT CONFIGS #######
#########################

# Import these at the top if you want. they would be used as app.names
# We've imported them here for easy reference
from beanftsite.core.views import core
from beanftsite.users.views import users
from beanftsite.uploader.views import uploader
from beanftsite.error_pages.handlers import error_pages

# Register the apps
app.register_blueprint(users)
app.register_blueprint(uploader)
app.register_blueprint(core)
app.register_blueprint(error_pages)
