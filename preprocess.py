import numpy as np

trainSet = 'train_clean.txt'
testSet = 'test_clean.txt'
relationSet = 'relation.txt'

# 把标签(关系类型)转换成int类型
label_int = {}
int_label = {}
f_relation = open(relationSet, 'r')
for line in f_relation:
    l = line.strip().split()
    index = int(l[1])
    relation = str(l[0])
    label_int[relation] = index
    int_label[index] = relation
# print(label_int)
# print(int_label)

# 建立wordSet
wordSet = set()
files = [trainSet, testSet]
for f_name in files:
    f = open(f_name, 'r')
    lines = f.readlines()
    f.close()
    for l in lines:
        lines_clean = l.strip().split(" ")[2:]
        for word in lines_clean:
            wordSet.add(word)
# print("len(wordSet)", len(wordSet))
f_word = open('wordSet.txt', 'w')
for word in sorted(wordSet):
    f_word.write(str(word) + "\n")
# print('wordSet.txt created')

# word embeddings
# 这一段copy的 大概就是用GoogleNews-vectors-negative300进行w2v
emb_google_txt = './word_embeddings/GoogleNews-vectors-negative300.txt'
avg_vec_file = './word_embeddings/GoogleNews-vectors-negative300_avg_vec.txt'
word_to_emb = {}
with open(emb_google_txt, 'r', encoding='utf-8') as f:
    first = True
    for line in f:
        if first == True:
            first = False
            continue
        line = line.strip().split()
        if len(line) != 301:
            continue
        word = str(line[0])
        vec = [float(x) for x in line[1:]]
        vec = np.array(vec, dtype='float64')
        if word in wordSet:
            word_to_emb[word] = vec
        elif word.lower() in wordSet and word.lower() not in word_to_emb:
            word_to_emb[word.lower()] = vec


def get_avg_vec(file_name):
    with open(file_name, 'r') as f:
        line = f.readline()
        line = line.strip().split()
        line = [float(x) for x in line]
        avg_vec = np.array(line, dtype='float64')
        print("avg_vec.shape", avg_vec.shape)
        return avg_vec


avg_vec = get_avg_vec(avg_vec_file)

word_to_int = {}
embedding = []
unknown_words = 0
word_to_int['PADDING'] = len(word_to_int)
embedding.append(np.zeros(300, dtype='float64'))

for w in sorted(wordSet):
    word_to_int[w] = len(word_to_int)
    if w in word_to_emb:
        embedding.append(word_to_emb[w])
    elif w.lower() in word_to_emb:
        embedding.append(word_to_emb[w.lower()])
    else:
        unknown_words += 1
        embedding.append(avg_vec)

embedding = np.array(embedding, dtype='float64')
print("len(word_to_int)", len(word_to_int))  # 25656
print("embedding.shape", embedding.shape)  # (25656, 300)
print("unknown_words", unknown_words)  # 2652
# copy完毕。。。

# 获得最长句长
def get_max_sent_len(files):
    max_sent_len = 0
    for fname in files:
        f = open(fname, 'r')
        lines = f.readlines()
        f.close()
        for l in lines:
            l = l.strip().split(" ")[2:]
            max_sent_len = max(max_sent_len, len(l))
    return max_sent_len


max_sent_len = get_max_sent_len(files)
print("max_sent_len", max_sent_len) # 102-1 = 101


# 建造可以输入model_data.npy
def create_matrices(file_name, word_to_int, label_to_int, max_sent_len):
    X = []
    Y = []

    f = open(file_name, 'r')
    lines = f.readlines()
    f.close()
    lines = [line.strip().split() for line in lines]

    for line in lines:
        Y.append(label_to_int[line[1]])
        line = line[2:]
        tmp = np.zeros(max_sent_len, dtype='int32')
        for i in range(len(line)):
            tmp[i] = word_to_int[line[i]]
        X.append(tmp)

    X = np.array(X, dtype='int32')
    Y = np.array(Y, dtype='int32')

    return [X, Y]


train_set = create_matrices(trainSet, word_to_int, label_int, max_sent_len)
test_set = create_matrices(testSet, word_to_int, label_int, max_sent_len)
model_data = [train_set, test_set, embedding, label_int, int_label]
np.save('model_data', model_data)
