# create a virtual env using CMD --------------------------
virtualenv env

# activate the activate file 
env\Scripts\activate

# install the needed 2 dependencies
pip install flask pandas

# generate the req files
pip freeze > requirements.txt

touch run.py



# docker compose ----------------------------
# docker-compose up doesn't actually rebuild images, instead use 
# should work !?
docker-compose up --force-recreate

# modules ----------------------------------
Example
Save this code in a file named mymodule.py

def greeting(name):
  print("Hello, " + name)
Use a Module
Now we can use the module we just created, by using the import statement:

Example
Import the module named mymodule, and call the greeting function:

import mymodule

mymodule.greeting("Jonathan")
""Note: When using a function from a module, use the syntax: module_name.function_name.""



## Github CI/CD

![ci-cd-flow-desktop_1](D:\Desktop\github_repo\prod_beanft_ml_api\ci-cd-flow-desktop_1.png)
