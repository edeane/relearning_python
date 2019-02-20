"""
Relearning the basics after being away from python for a year!

relearing packages order:
X pycharm stuff (venv, shortcuts, https://www.jetbrains.com/help/pycharm/meet-pycharm.html)
X python tutorial
X python standard library
X mysqlclient pandas real_sql to_sql
X unittests (https://www.jetbrains.com/help/pycharm/testing-your-first-python-application.html)
- numpy
- pandas
- plotting
- threading, parallel, multiprocessing
- creating modules and packages (https://docs.python.org/3/distributing/index.html)
- blackjack project
- ncaabb project
- django
- magic methods
- decorators (property, classmethod, staticmethod)
- more standard library https://docs.python.org/3/library/index.html

Pycharm setup setting
- Editor > Font > Size: 13
- Keymap > Execute > Execute selection in console: Ctrl + Enter
- Tools > Python External Documentation: https://stackoverflow.com/questions/49777474/does-anyone-have-the-
    documentation-urls-in-pycharm-for-the-following-libraries
- Install  https://www.lfd.uci.edu/~gohlke/pythonlibs/#mysql-python mysqlclient
- pip install C:/Users/<username> /Downloads/mysqlclient-1.3.13-cp37-cp37m-win_amd64.whl
- Check Settings > Appearance & Behavior > Widescreen tool window layout
- Pycharm documentation ctrl + q for documentation

Official Python Documentation https://docs.python.org/3/
- Tutorial https://docs.python.org/3/tutorial/index.html
- Standard Library https://docs.python.org/3/library/index.html
- Language Reference https://docs.python.org/3/reference/index.html



create requirements.txt
- pip freeze > requirements.txt

learn about documentation tools
- https://matplotlib.org/sampledoc/
- http://www.sphinx-doc.org/en/master/
- http://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html
- https://www.mkdocs.org/
- Google Docstring Standard https://github.com/google/styleguide/blob/gh-pages/pyguide.md

learn about project folder structure
- https://drivendata.github.io/cookiecutter-data-science/ talk: http://isaacslavitt.com/2016/07/20/data-science-
    is-software-talk/
- https://blog.godatadriven.com/how-to-start-a-data-science-project-in-python

"""

# CTRL + Q for documentation
print('hello')

tasks = {'completed': 0, 'name': 'this new motion', 'ed': 'deane'}
# Basic completion CTRL + Space once
print(tasks['ed'])

# CTRL + Space twice
# sock

# Smart Completion = CTRL + Shift + Space
def f(x):
    x.append(10)
    x.remove(10)
    return x


class Dog:
    def __init__(self, name, breed):
        self.name = name
        self.breed = breed
        self.sit_pos = False

    def run(self, miles):
        for i in range(miles):
            print('ran', i, 'mile')
        return miles ** 2

    def __repr__(self):
        return self.name

    def do_sit(self):
        self.sit_pos = not self.sit_pos

    def is_sit(self):
        if self.sit_pos == True:
            position = 'sitting'
        elif self.sit_pos == False:
            position = 'standing'
        return position








if __name__ == '__main__':
    print('this will not run when the module is imported')

    a_dog = Dog('bud', 'lab')
    print(a_dog)
    run_ret = a_dog.run(10)
    print('run:', run_ret)
    print(a_dog.is_sit())
    a_dog.do_sit()
    print(a_dog.is_sit())
    a_dog.do_sit()
    print(a_dog.is_sit())




