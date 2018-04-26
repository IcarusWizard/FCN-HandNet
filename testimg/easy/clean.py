import os, re

if __name__ == '__main__':
    search = re.compile(r'Flooded_(\d*).png')
    for name in filter(lambda name:search.search(name), os.listdir()):
        os.system('del ' + name)