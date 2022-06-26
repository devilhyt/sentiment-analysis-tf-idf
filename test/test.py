with open('./data/' + "happy" + ".txt", 'r', encoding="utf-8") as f:
    content_list = [line.rstrip('\n') for line in f]
cnt = 0
for c in content_list:
    if len(c) > 512:
        cnt+=1
print(cnt)
