
def greeting(name='there'):
    return f'Hello, {name}'

def get_id(id):
    ids = {
        123: 'ab',
        456: 'cd',
        789: 'ef',
        101: 'gh'
    }
    return f'id: {id}, name: {ids.get(id, "error")}'

def fact(n):
    return 1 if n == 1 else n * fact(n-1)


print(f'This is the fact of 20: {fact(20)} that ran when mod1 is imported')

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        print(fact(int(sys.argv[1])))





