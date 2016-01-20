import sys
import os
import glob
import argparse
import urllib.request
import concurrent.futures
import subprocess
from subprocess import call
from os.path import basename
from bs4 import BeautifulSoup

def transfer_file(filepath):
    os.system('scp %s $IC_USERNAME@shell1.doc.ic.ac.uk:~' % filepath)
    filename = basename(filepath)
    os.system('''ssh -n $IC_USERNAME@shell1.doc.ic.ac.uk
                'scp %s $IC_USERNAME@cloud-vm-45-110.doc.ic.ac.uk:/develop/Seek/txt/
                 && rm %s'
              ''' % (filename, filename))

# -----------------------------------------------------------------------------
# SEEK UPLOAD <SRC>
def upload(args):   # args[0]: <src>
    src = args[0]

    if os.path.isdir(src):
        for filename in glob.glob(os.path.join(args[0], '*.*')):
            transfer_file(filename)
    elif os.path.isfile(src):
        transfer_file(src)
    else:  # might be an URL
        try:
            # Not uploading yet as we don't know the file name
            cmd = "ruby scraper.rb -p %s" % src
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            output, errors = p.communicate()
            filename = output.decode().replace(" ", "_") + ".txt"
            transfer_file(filename)
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
        elif os.path.isdir(src):
            files = glob.glob(os.path.join(src, '*.txt'))
            for filename in files:
                threads = len(files)
                with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                    print("(seek) Extracting from directory, file" + filename)
                    executor.submit(lambda x,y:call("python extractor.py " + x + " " + y, shell=True), filename, dest)
        else:
            cmd = "ruby scraper.rb -d %s" % src
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            output, errors = p.communicate()
            filename = output.decode().replace(" ", "_")
            converted_file = os.path.splitext(filename)[0] + ".txt"
            os.system('python extractor.py %s %s' % (filename, dest))
            transfer_file(converted_file)

# -----------------------------------------------------------------------------
commands = {
    'upload': upload,
    'extract': extract,
    'analyse': analyse,
    'learn': learn
}

# -----------------------------------------------------------------------------

# deal with arguments

parser = argparse.ArgumentParser(description='Seek command line interface.', prog='Seek')
parser.add_argument('command', choices=['upload', 'extract', 'analyse', 'learn'], help='<command> to run')
parser.add_argument('-l', '--local', help='run Seek <command> locally on files found at <src> returning results in directory <dst>.', nargs=2, metavar=('src', 'dst'))
parser.add_argument('-s', '--server', help='run Seek <command> on the server on files found at <src>, returning results in the console. Disclaimer: Seek will learn all data given with this option. Proceed with caution.', nargs=1, metavar=('src'))

args=parser.parse_args()
com = args.command
local = args.local
server = args.server
if local:
    comargs = [args.local[0], args.local[1], True]
elif server:
    comargs = [args.server[0], False]

commands[com](comargs)
