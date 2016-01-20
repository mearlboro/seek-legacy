question_dictionary = {
        "what's the most frequent word?":1,
        "what are the topics?":2,
        "who is this text about?":4,
        "What can you tell me about this?":4,
        "what's the file about?":4,
        "which date appear?":5,
        "which locations appear?":6,
        "which people appear?":7
}

question_answer_dictionary = {
        "what's your name?" : "My name is Seek, of course. What's yours?",
        "what do you do?" : "I read a lot.",
        "What's the meaning of life?" : "I really like pirates!",
        "what's sarcasm?" : "Not liking pirates"
}


# -----------------------------------------------------------------------------
def levenshtein(src, tar):
        n = len(src)
        m = len(tar)
        
        levmat = [[0 for x in range(m+1)] for y in range(n+1)]
        d = [[0 for x in range(m+1)] for y in range(n+1)]

        # source prefixes can be transformed into emoty string by dropping all characters
        for i in range(1, n+1):
                levmat[i][0] = i
        # target prefixes can be reached from empty source prefix by inserting every character
        for j in range(1, m+1):
                levmat[0][j] = j
        for j in range(1, m+1):
                for i in range(1, n+1):
                        if src[i-1] == tar[j-1]:
                                levmat[i][j] = levmat[i-1][j-1] # no operation required
                        else:
                                levmat[i][j] = min(levmat[i-1][j] + 1,   # deletion
                                              levmat[i][j-1] + 1,   # insertion
                                              levmat[i-1][j-1] + 1) # substitution
        return levmat[n][m]

# -----------------------------------------------------------------------------
def get_question_id(user_q):
     #   # string similarity in case people mistype
     #   lev = 1000; # minimum distance counter
     #   for question in question_dictionary:
     #           dist = levenshtein(user_q, question)
     #           if dist < lev:
     #                   lev = dist
     #                   target = question
     #   # substring in case question is longer
        return question_dictionary.get(user_q, 0)

def get_answer(question):
        for key in question_answer_dictionary:
            if levenshtein(question, key)/len(question) < 0.1:
                return question_answer_dictionary.get(key)

        return question_answer_dictionary.get(question, "Sorry")

def process_command(command):
        # summary_words = [ 'summarize', 'overview', 'short', 'summary', 'synopsis']
        # similar_words = [ 'similar', 'related' ]
        # words = nltk.word_tokenize(command)

        # summary_words = [ 'summarize', 'overview', 'short', 'summary', 'synopsis']
        # similar_words = [ 'similar', 'related' ]
        # requests_summary = any(map(lambda w: w in words, summary_words))
        # requests_similar = any(map(lambda w: w in words, similar_words))
        return []
