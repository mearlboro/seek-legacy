# This script requires the textract package,
# which you can download following http://textract.readthedocs.org/en/latest/installation.html
# When installing (on Linux) make sure to remove ffmpeg from command line
import os
import sys
import textract

# Use script by calling $ python extractor.py <source> <destination>
# By default destination folder is set to /deep-shit/Raw Text

if len(sys.argv) == 1 : sys.exit(0)
dst = os.getcwd()[:-len('File Processor')] + 'Raw Text'
if len(sys.argv) > 1 :
    src = sys.argv[1]
    if len(sys.argv) > 2 :
        dst = sys.argv[2]

for root, dirs, files in os.walk(src, True) :
    print 'stepped into ' + root
    for name in files :
        filename, file_ext = os.path.splitext(name)
        text = textract.process(os.path.join(root, name))
        f = open(dst + '/' + filename + '.txt', 'w+')
        f.write(text)
        f.close()
