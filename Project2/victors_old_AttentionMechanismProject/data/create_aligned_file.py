def addToList(i,words):
    #if it is not the last word, add before next words appearance
    if i != len(data_list)-1 :
        #if the next word is in the list
        if (i+1) in words:
            nextI = words.index(i+1)
            words.insert(nextI, i)
        #else add the next word and add this word
        else :
            words = addToList(i+1,words)
            words = addToList(i,words)
    #else place at end
    else :
        words.append(i)
    return words


    

#the file with alignment src-tar
align = open('aligned.grow-diag-final-and','r')

#file with src sentences
dataSrc = open('low/kyoto-train.cln.ja','r')

#created file with alignment of words
aligned  = open('kyoto-train.align','w')

#for each sentence create alignment
for line in dataSrc:
    #list for all the words
    data_list = []

    #list for given src-tar alignments
    align_list = []

    #new alignment
    aligned_list = []

    #to test if al src words are used
    used_words = []

    #tokenize and place all words in the list
    for word in line.split():
        data_list.append(word)
    
    #read align pairs and take src alignment
    lineAl = align.readline()
    for word in lineAl.split():
        tmp = word.split('-',1)
        nextWord = int(tmp[0])

        #if a src word is used multiple times, place only at first apearance 
        if nextWord not in used_words:
            aligned_list.append(data_list[nextWord])
            used_words.append(nextWord)
   
   #test if al src words are used
    for i in xrange(0,len(data_list)):
        if i not in used_words:
            used_words = addToList(i,used_words)
    
    #write alignment for this sentence to the file
    for item in used_words:
        aligned.write(str(item)+' ')
    aligned.write('\n')

