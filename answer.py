# 创建只有序号和label的dataset
def createAnswer(inputPath, outputPath):
    fIn = open(inputPath, 'r')
    lines = fIn.readlines()
    fOut = open(outputPath, 'w')
    for i in range(0, len(lines)):
        l = lines[i].strip().split(" ")
        num = str(i + 1)
        label = str(l[1])
        fOut.write(num + "\t" + label)
        fOut.write("\n")
    # print(outputPath + " " + "Created")


createAnswer('train_clean.txt', 'train_answer.txt')
createAnswer('test_clean.txt', 'test_answer.txt')