import logging
from gensim import corpora, models, similarities
from pprint import pprint   # pretty-printer
import os
import glob as glob
from operator import itemgetter

# Uncomment for logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

directory = os.chdir("../raw/text/")
files = glob.glob("*.txt")
documents = []

# Add the content of each text file as an element in the list
for f in files:
    open_f = open(f, 'r')
    documents.append(open_f.read())
    open_f.close()

separators = """a
about
above
after
again
against
all
am
an
and
any
are
aren\'t
as
at
be
because
been
before
being
below
between
both
but
by
can
can\'t
cannot
could
couldn\'t
did
didn\'t
do
does
doesn\'t
doing
don\'t
down
during
each
few
for
from
further
had
hadn\'t
has
hasn\'t
have
haven\'t
having
he
he\'d
he\'ll
he\'s
her
here
here\'s
hers
herself
him
himself
his
how
how\'s
i
i\'d
i\'ll
i\'m
i\'ve
if
in
into
is
isn\'t
it
it\'s
its
itself
let\'s
me
more
most
mustn\'t
my
myself
no
nor
not
of
off
on
once
only
or
other
ought
our
ours
ourselves
out
over
own
same
shan\'t
she
she\'d
she\'ll
she\'s
should
shouldn\'t
so
some
such
than
that
that\'s
the
their
theirs
them
themselves
then
there
there\'s
these
they
they\'d
they\'ll
they\'re
they\'ve
this
those
through
to
too
under
until
up
very
was
wasn\'t
we
we\'d
we\'ll
we\'re
we\'ve
were
weren\'t
what
what\'s
when
when\'s
where
where\'s
which
while
who
who\'s
whom
why
why\'s
with
won\'t
would
wouldn\'t
you
you\'d
you\'ll
you\'re
you\'ve
your
yours
yourself
yourselves
•
\\x20
\\n
( ) , . ! ? = + - _ / * ^
et
al
using
data
$2/>"""
stoplist = set(separators.split())

# Remove stoplist elements
texts = [[word for word in document.lower().split() if word not in stoplist]
        for document in documents]

# Remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
 for token in text:
     frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
      for text in texts]

# pprint(texts)

# How many times does each word appear in the document"
dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/seek.dict') # store the dictionary, for future reference
# print(dictionary)

# Prints mapping between words and their ids
# print(dictionary.token2id)

# Doc2Bow counts the number of occurences of every distinct word, converts the
# word to its id and maps it to the count
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/seek.mm', corpus) # store to disk, for later use

# Get most frequent word
# maxV = 0
# for vector in corpus:
#     if vector:
#         maxV = max(maxV, max(vector, key=itemgetter(1))[0])
    # print(vector)
# print(dictionary[maxV])

tfidf = models.TfidfModel(corpus)

# Transformed (normalized?) corpus
corpus_tfidf = tfidf[corpus]

# Print transformed corpus
# for doc in corpus_tfidf:
#     print(doc)

# Testing for lsi
# lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=127) # initialize an LSI transformation
# corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
# pprint(lsi.print_topics(127))

lda = models.LdaModel(corpus, id2word=dictionary, num_topics=127)
pprint(lda.print_topics(127))





################################################################################
# Old Stuff
# lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=100) # initialize an LSI transformation
# corpus_lsi = lsi[corpus_tfidf] # c
# lsi.print_topics(300)
# corpus_lda = lda[corpus_tfidf]
# for doc in corpus_lda: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
#     print(doc)

# output = open("results.txt", "w+")
# output.write(str(lda.print_topics(100)))
