'''
Relearning the basics after being away from python for a year.

1)
Pycharm setup setting

- Editor > Font > Size: 13
- Keymap > Execute > Execute selection in console: Ctrl + Enter
- Tools > Python External Documentation: https://stackoverflow.com/questions/49777474/does-anyone-have-the-
    documentation-urls-in-pycharm-for-the-following-libraries
- Ctrl + Q view documentation then pin
- Install  https://www.lfd.uci.edu/~gohlke/pythonlibs/#mysql-python mysqlclient
- pip install C:/Users/<username> /Downloads/mysqlclient-1.3.13-cp37-cp37m-win_amd64.whl


Official Python Documentation https://docs.python.org/3/
- Tutorial https://docs.python.org/3/tutorial/index.html
- Standard Library https://docs.python.org/3/library/index.html
- Language Reference https://docs.python.org/3/reference/index.html


ctrl + q for documentation
datatypes (string, int, double) and
container datatypes (dict, list, set, tuple)
collections datatypes (namedtuple, deque, ChainMap, Counter, OrderedDict, defaultdict, UserDict, UserList, UserString)
loops

create requirements.txt
pip freeze > requirements.txt


learn about documentation tools
https://matplotlib.org/sampledoc/
http://www.sphinx-doc.org/en/master/
http://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html
https://www.mkdocs.org/


Google Docstring Standard
https://github.com/google/styleguide/blob/gh-pages/pyguide.md

pylint
https://www.pylint.org/

pep 8
https://www.python.org/dev/peps/pep-0008/
http://flake8.pycqa.org/en/latest/index.html


learn about project folder structure
https://drivendata.github.io/cookiecutter-data-science/ talk: http://isaacslavitt.com/2016/07/20/data-science-
    is-software-talk/
https://blog.godatadriven.com/how-to-start-a-data-science-project-in-python

'''



def add_together(x, y):
    '''
    This function adds together two integers

    Args:
        x:
        y:

    Returns:

    '''

    print(x, ' ', y)

    return x+y




if __name__ == '__main__':

    xy = add_together(2, 3)
    print('x + y equals: ', xy)

























