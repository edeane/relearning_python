
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup


def simple_get(url):
    """Attempts to get the content at the provided url by making an HTTP GET request.

    Args:
        url(str): string url

    Returns:
        If the content-type of response is some kind of HTML/XML, return the text content, otherwise return None.

    Raises:
        RequestException: If there is an error during the request.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        # log_error(str(e))
        log_error(f'Error during requests to {url} :\n{e}')
        return None


def is_good_response(resp):
    """Determines if the response seems to be HTML.

    Args:
        resp: The response from the GET request.

    Returns:
        bool: True if HTML, False otherwise
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)


def log_error(e):
    """
    Args:
        e:

    Returns:
    """
    print(e)


def get_names():
    """Downloads the Top 100 Greatest Mathematicians of the Past

    Returns:
        list: list of strings of each mathematician

    Raises:
        Exception if it cannot get info from url
    """
    raw_math_url = 'http://www.fabpedigree.com/james/mathmen.htm'
    raw_math_html = simple_get(raw_math_url)

    if raw_math_html is not None:
        math_html = BeautifulSoup(raw_math_html, 'html.parser')
        names = []
        for i in math_html.select('li'):
            names.append(i.text.split('\n')[0].strip())

        return (names)

    raise Exception(f'Error retrieving contents at {raw_math_url}')

