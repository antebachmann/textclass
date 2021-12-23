



tweet_list=[]
with open('data/train_pos.txt','r') as fh:
    for line in fh:
        tweet_list.append(line)
print(tweet_list[2])