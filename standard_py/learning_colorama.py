"""

Learning colorama

https://pypi.org/project/colorama/
https://github.com/tartley/colorama

"""


from colorama import init
from colorama import Fore, Back, Style

# Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# Style: DIM, NORMAL, BRIGHT, RESET_ALL

def print_color(x, font='red', back='none'):
    """
    black, red, green, yellow, blue, magenta, cyan, or white
    """
    font_dict = {'black': Fore.BLACK,
                 'red': Fore.RED,
                 'green': Fore.GREEN,
                 'yellow': Fore.YELLOW,
                 'blue': Fore.BLUE,
                 'magenta': Fore.MAGENTA,
                 'cyan': Fore.CYAN,
                 'white': Fore.WHITE,
                 'none': ''
                 }
    back_dict = {'black': Back.BLACK,
                 'red': Back.RED,
                 'green': Back.GREEN,
                 'yellow': Back.YELLOW,
                 'blue': Back.BLUE,
                 'magenta': Back.MAGENTA,
                 'cyan': Back.CYAN,
                 'white': Back.WHITE,
                 'none': ''
                 }
    print(font_dict.get(font, Fore.RED) + back_dict.get(back, '') + x + Style.RESET_ALL)
    return None





