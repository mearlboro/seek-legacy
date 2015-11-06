import os
import sys
import textract

# Use script by calling $ python extractor.py <source> <destination>
# By default destination folder is set to /deep-shit/Raw Text

if len(sys.argv) == 1 : sys.exit(0)
dst = os.getcwd()[:-len('seek-env')] + 'txt'
if len(sys.argv) > 1 :
    src = sys.argv[1]
    if len(sys.argv) > 2 :
        dst = sys.argv[2]

filepath, file_ext = os.path.splitext(src)
filename = os.path.basename(filepath)
try:
    text = textract.process(src)
except TypeError:
    # do nothing
print "Converting " + filename
f = open(dst + '/' + filename + '.txt', 'w+')
f.write(text)
f.close()
