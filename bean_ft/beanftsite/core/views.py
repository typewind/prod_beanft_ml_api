from flask import render_template,request,Blueprint
from beanftsite.models import FileContents

core = Blueprint('core',__name__)

@core.route('/')
def index():

    return render_template('index.html')


