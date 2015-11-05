# This script requires the textract package,
# which you can download following http://textract.readthedocs.org/en/latest/installation.html
# When installing (on Linux) make sure to remove ffmpeg from command line
import os
import sys
import textract
import glob
# Use script by calling $ python extractor.py <source> <destination>
# By default destination folder is set to /deep-shit/Raw Text

if len(sys.argv) == 1 : sys.exit(0)
dst = os.getcwd()[:-len('File Processor')] + 'Raw Text'
if len(sys.argv) > 1 :
    src = sys.argv[1]
    if len(sys.argv) > 2 :
        dst = sys.argv[2]

os.chdir(src)
for root, dirs, files in os.walk('.', True) :
    print 'stepped into ' + root

    for subdirname in dirs:
            print(os.path.join(root, subdirname))

    # print path to all filenames.
    for filename in files:
        path = os.path.join(root, filename)
        name, file_ext = os.path.splitext(filename)
        text = textract.process(path)
        print "Converting " + filename
        f = open(dst + '/' + name + '.txt', 'w')
        f.write(text)
        f.close()

    # for name in files :
    #
