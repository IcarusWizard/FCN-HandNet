import os, re

def clean(search, path = '.\\'):
    for name in filter(lambda name: search.search(name), os.listdir(path)):
        os.system('del ' + os.path.join(path, name))

if __name__ == '__main__':
    to_del = [r'Flooded_(\d*).png', r'Test_(\d*)..*']
    for search in [re.compile(r) for r in to_del]:
        clean(search)