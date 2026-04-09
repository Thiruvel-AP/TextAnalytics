# Custom stop words 
custom_stopwords = frozenset([
    'please', 'help', 'thank', 'thanks', 'hi', 'hello',
    'dear', 'regards', 'appreciate', 'kindly',
    'issue', 'problem', 'assist', 'experiencing', 'need',
    'having', 'facing', 'would', 'could', 'get', 'got',
    'also', 'using', 'use', 'let', 'know', 'like', 'one',
    'set', 'add',
    'try', 'tried', 'trying', 'want', 'wanted',
    'work', 'working', 'resolve', 'resolved',
    'look', 'looking', 'go', 'going', 'keep',
    'take', 'make', 'making', 'able', 'unable',
    'receive', 'received', 'provide', 'provided',
    'product', 'customer', 'service', 'support',
    'team', 'response', 'request', 'information',
    'time', 'day', 'week',
    'since', 'still', 'even', 'much',
    'really', 'already', 'just', 'yet',
    'update', 'device', 'data', 'software', 'account',
    'step', 'persists', 'perform', 'troubleshoot', 'contact',
    'start', 'notice', 'fine', 'check', 'multiple',
    'available', 'productivity', 'reset', 'option',
    'change', 'say', 'possible', 'specific', 'action', "step",
])

# tag map for the lemmatization
tag_map = {'J': 'a', 'V': 'v', 'N': 'n', 'R': 'r'}