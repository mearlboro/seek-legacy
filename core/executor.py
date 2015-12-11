import sys
import os
import glob
import argparse
import urllib.request
import concurrent.futures
from subprocess import call
from bs4 import BeautifulSoup


# -----------------------------------------------------------------------------
# SEEK UPLOAD <SRC>
def upload(args):   # args[0]: <src>
    src = args[0]
    
    if os.path.isdir(src):
        for filename in glob.glob(os.path.join(args[0], '*.*')):
            print("TODO: upload this file to the raw/ folder on the server and flag it that it needs to be extracted");
    elif os.path.isfile(src):
        print("TODO: upload this file to the raw/ folder on the server and flag it that it needs to be extracted");
    else:  # might be an URL
        try:
            # TODO(dd2713): add a call to the updated scraper
            with urllib.request.urlopen(args[0]) as response: # if it is a url, grab the html
                html = response.read()
                content = BeautifulSoup(html, "lxml")
                print(content);
                title = content.find('title').text
                f = open('/raw/html/' + title + '.html', 'w+')
                f.write(html)
                f.close()
        except IOError:
            print("(seek) Cannot parse this address or URL.")


# -----------------------------------------------------------------------------
# SEEK EXTRACT [--local] <SCR> <DEST> 
def extract(args):  # args[0]: <src>, args[1]: <dest>, args[2]: isLocal
    src = args[0]
    dest = args[1]
    loc = args[2]

    if loc: 
        # if it's a file, just run extractor
        if os.path.isfile(src): 
            print("(seek) Extracting file: " + src)
            retcode = call("python extractor.py " + src + " " +  dest, shell=True)
            if retcode < 0:
                print("(seek) Call to extractor terminated by signal: " + -retcode)

        # if its a directory, grab all files and extract concurrently        
        else: 
            files = glob.glob(os.path.join(src, '*.txt'))
            for filename in files:
                threads = len(files)
                with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                    print("(seek) Extracting from directory, file" + filename)
                    executor.submit(lambda x,y:call("python extractor.py "+x+" "+y, shell=True), filename, dest)
        # TODO: if it is an URL to a file?
    else:
        print("TODO: upload these files to the txt/ folder on the server and flag it that it needs to be analysed");


# -----------------------------------------------------------------------------
# SEEK ANALYSE [--local] <SRC> <DEST>
def analyse(args):  # args[0]: <src>, args[1]: <dest>, args[2]: isLocal
    print("TODO: topic modelling")


# -----------------------------------------------------------------------------
# SEEK LEARN [--local]
def learn(args):  # args[0]: <src>, args[1]: <dest>, args[2]: isLocal
    print("TODO: machine learning")


# -----------------------------------------------------------------------------
commands = {
    'upload': upload,
    'extract': extract,
    'analyse': analyse,
    'learn': learn
}

# -----------------------------------------------------------------------------
if len(sys.argv) <= 3:
    print("Seek expects the following command \n seek <command> [<flags>] <src> <dest>")
    sys.exit(0)
if len(sys.argv) > 3:
    com  = sys.argv[1]
    src  = sys.argv[2]
    dest = sys.argv[3]
    if commands.get(com, False):
        commands[com]([src, dest, True])
    else:
        print("<command> can be \n upload \n extract \n analyse \n learn")
        sys.exit(0)

