import os

os.environ['CLASSPATH'] = "D:/stanford-postagger-full-2017-06-09"
from nltk.tokenize.stanford import StanfordTokenizer

java_path = "C:/Program Files/Java/jdk-13.0.1/bin/java.exe"
os.environ['JAVAHOME'] = java_path

trainSet = 'TRAIN_FILE.TXT'
testSet = 'TEST_FILE_FULL.TXT'

outputTrain = 'train_clean.txt'
outputTest = 'test_clean.txt'

def dataClean(inputPath, outputPath):
    lines = [line.strip() for line in open(inputPath)]  # 存储数据集中每一行
    fOut = open(outputPath, 'w', encoding="utf-8")
    for i in range(0, len(lines), 4):  # 根据训练集，每四行是一个测试句子
        num = lines[i].split("\t")[0]  # 取出序号
        sentence = lines[i].split("\t")[1][1:-1]  # 取出句子，并去掉引号
        label = lines[i + 1]  # 取出标签
        sentence = sentence.replace("<e1>", " E1S ")  # 替换<e1>
        sentence = sentence.replace("</e1>", " E1E ")
        sentence = sentence.replace("<e2>", " E2S ")
        sentence = sentence.replace("</e2>", " E2E ")
        tokens = StanfordTokenizer().tokenize(sentence)  # 分词
        fOut.write(" ".join([num, label, " ".join(tokens)]))  # 把序号标签放前面，连词成句
        fOut.write("\n")


dataClean(trainSet, outputTrain)
dataClean(testSet, outputTest)

print('data clean finished')



