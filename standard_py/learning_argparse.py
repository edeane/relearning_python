"""Learning the argparse standard library.

How to Build Command Line Interfaces in Python With argparse
https://realpython.com/command-line-interfaces-python-argparse/

1. Import the Python argparse library
2. Create the parser
3. Add optional and positional arguments to the parser
4. Execute .parse_args()

"""


import argparse

class VerboseStore(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError('nargs not allowed')
        super(VerboseStore, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print(f'Here I am, setting the values {values}, for the {option_string} option...')
        setattr(namespace, self.dest, values)


my_parse = argparse.ArgumentParser()
my_parse.version = '1.0.1'
my_parse.add_argument('-a', action='store', help='a store default', nargs=3, type=int)
my_parse.add_argument('-b', action='store_const', const=42, help='b store_const default')
my_parse.add_argument('-c', action='store_true', help='c store_true optional')
my_parse.add_argument('-d', action='store_false')
my_parse.add_argument('-e', action='append')
my_parse.add_argument('-f', action='append_const', const=43)
my_parse.add_argument('-g', action='count')
my_parse.add_argument('-i', '--input', action=VerboseStore, type=int)
my_parse.add_argument('-j', action='version')
my_parse.add_argument('-k', action='append', help='a store default', nargs=3)
my_parse.add_argument('-l', action='store', nargs='*', default='my default value')
my_parse.add_argument('-m', action='store', choices=range(0, 5), required=False, type=int)

# my_group = my_parse.add_mutually_exclusive_group(required=True)
# my_group.add_argument('-v', '--verbose', action='store_true')
# my_group.add_argument('-s', '--silent', action='store_true')

my_parse.add_argument('-v', '--verbosity', action='store', type=int, dest='my_verbosity_level', metavar='LEVEL')

args = my_parse.parse_args()

from pprint import pprint
pprint(vars(args))

def run_file():
    runfile('/relearning_python/standard_py/learning_argparse.py',
            args=['1', '-b', '-c', '-d', '-e', '422', '-e', '500', '-e', 'eddie', '-f', '-f', '-fff', '-gggggggg'],
            wdir='/relearning_python/standard_py')

    runfile('/relearning_python/standard_py/learning_argparse.py',
            args=['-a', '42', '42', '123', '-i', '422'],
            wdir='/relearning_python/standard_py')

    runfile('/relearning_python/standard_py/learning_argparse.py',
            args=['-i', '422'],
            wdir='/relearning_python/standard_py')

    runfile('/relearning_python/standard_py/learning_argparse.py',
            args=['-k', '1', '2', '3', '-k', '4', '5', '6'],
            wdir='/relearning_python/standard_py')

    runfile('/relearning_python/standard_py/learning_argparse.py',
            args=['-j'],
            wdir='/relearning_python/standard_py')

    runfile('/relearning_python/standard_py/learning_argparse.py',
            args=['-l', 'my input value'],
            wdir='/relearning_python/standard_py')

    runfile('/relearning_python/standard_py/learning_argparse.py',
            args=[],
            wdir='/relearning_python/standard_py')

    runfile('/relearning_python/standard_py/learning_argparse.py',
            args=['-a', '123', '456', '789'],
            wdir='/relearning_python/standard_py')

    runfile('/relearning_python/standard_py/learning_argparse.py',
            args=['-m', '2'],
            wdir='/relearning_python/standard_py')
