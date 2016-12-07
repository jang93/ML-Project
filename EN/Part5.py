
# coding: utf-8

# In[ ]:

# Part 5
import codecs # to write to file
import pickle # to store parameter dictionaries
from subprocess import PIPE, run # to run cmdline commands
import matplotlib.pyplot as plt # to plot graph
import copy # to create a copy

train = './train'
train_pre = './train_pre'
output_file = './train_out_'


def preprocessing(train_file, output_file, include_sentiment):
    with open(train_file, encoding='utf-8') as ifile, codecs.open(output_file, 'w', 'utf-8-sig') as ofile:
        for line in ifile:
            if len(line.split()) != 0:
                word = line.split()[0]
                # lower case all words
                word = word.lower()
                # Remove # & @ & ~ in front of word
                if (len(word))> 1 and (word[0] == "@" or word[0] == "#" or word[0] == "~"):
                    word = word[1:]
                    
                # Remove "'s" 
                if len(word)> 2 and "'" in word:
                    if word[-2:]== "'s":
                        word = word[:-2]
                    if word[-1:] == "'":
                        word = word[:-1]
                    if word[0] == "'":
                        word = word[1:]
             
                # Remove '"'
                if len(word)> 2 and '"' in word:
                    word = word.replace('"', '')
                    
                if include_sentiment:
                    sentiment = line.split()[1]
                    ofile.write(word + " " + sentiment+"\n")
                else:
                    ofile.write(word + "\n")
            else:
                ofile.write('\n')

                
                
        

# read train file & initialize parameters to 0
def initialize_params(train_file):
    with open(train_file, encoding = 'utf-8') as file:
        emission_count= {}
        transition_count= {}
        prev = 'START'
        end = 'STOP'
        transition_count[end] = {}
        for line in file:
            if len(line.split())!=0:
                pair = line.split()
                word = pair[0]
                sentiment = pair[1]   
                # emission params
                if word in emission_count.keys():
                    if sentiment in emission_count[word].keys():
                        pass
                    else:
                        sentiments = emission_count[word]
                        sentiments[sentiment] = 0
                else:
                    sentiment_count = {}
                    sentiment_count[sentiment] = 0
                    emission_count[word]=sentiment_count
                    
                # transition params
                if sentiment in transition_count.keys():
                    sentiment_list = transition_count[sentiment]
                    if prev in sentiment_list.keys():
                        pass
                    else:
                        sentiment_list[prev] = 0
                else:
                    new_sentiment = {}
                    new_sentiment[prev] = 0
                    transition_count[sentiment] = new_sentiment
                prev = sentiment
                
            else:
                sentiment_list = transition_count[end]
                if prev in sentiment_list.keys():
                    pass
                else:
                    sentiment_list[prev] = 0
                prev = 'START'
                    
        return (emission_count, transition_count)
    

def perceptron_algorithm(train_file, e_params, t_params):
    labels = ['B-neutral', 'I-neutral', 'B-negative', 'I-negative', 'B-positive', 'I-positive', 'O']
    with open(train_file, encoding ='utf-8') as ifile:
        unlabelled_sentence = []
        correct_sentence = []
        
        # Open train file and read line
        for line in ifile:
            if len(line.split())!=0:
                unlabelled_sentence.append(line.split()[0])
                correct_sentence.append(line.rstrip('\n'))
            else:
                # label sentence using params
                nodes = calculate_node_scores(unlabelled_sentence, e_params, t_params, labels)
                predicted_sentence = backtracking (unlabelled_sentence, nodes)
                
                # Compare predicted with correct and update params
                e_params, t_params = update_params(correct_sentence, predicted_sentence, e_params, t_params)
                
                # reset lists
                unlabelled_sentence = []
                correct_sentence = []
                
                
        return (e_params, t_params)

def calculate_node_scores(s, e_params, t_params, labels):
    nodes = {}
    #base case
    nodes[0] = {'START':[1,'nil']}
    #recursive
    for k in range (1, len(s)+1): #for each word
        X = s[k-1]
        for V in labels: #for each node
            prev_nodes_dict = nodes[k-1] #access prev nodes
            highest_score = 0
            parent = 'nil'
            
            # emission params
            if X in e_params.keys():
                e_labels = e_params[X]
                if V in e_labels.keys():
                    b = e_labels[V]
                else:
                    b = 0
            else:
                b = 0 
                
            # transition params
            for U in prev_nodes_dict.keys():
                prev_states = t_params[V]
                if U in prev_states.keys():
                    a = prev_states[U]
                else:
                    a = 0
                
                #prev node score
                prev_score = prev_nodes_dict[U][0]
                score = prev_score+a+b
                
                if score>= highest_score:
                    highest_score = score
                    parent = U
            if k in nodes.keys():
                nodes[k][V] = [highest_score,parent]
            else:
                new_dict = {V:[highest_score,parent]}
                nodes[k] = new_dict
            
    #end case
    prev_nodes_dict = nodes[len(s)]
    highest_score = 0
    parent = 'nil'
    for U in prev_nodes_dict.keys():
        #transition
        prev_states = t_params['STOP']
        if U in prev_states.keys():
            a = prev_states[U]
        else:
            a = 0
        #prev node score
        prev_score = prev_nodes_dict[U][0]
        score = prev_score+a
        if score>= highest_score:
            highest_score = score
            parent = U
    indiv_node = {'STOP': [highest_score,parent]}
    nodes[len(s)+1]=indiv_node
    
    return nodes


def backtracking(s, nodes):
    prev_state = 'STOP'
    for i in range(len(s)+1, 1,-1):
        prev_node = nodes[i][prev_state]
        prev_state = prev_node[1]
        s[i-2] += " "+prev_state
    return s
    
def update_params(correct, predicted, e_params, t_params):
    prev = 'START'
    end = 'STOP'
    
    # correct
    for centry in correct:
        pair = centry.split()
        word = pair[0]
        sentiment = pair[1]
        
        # update e_params
        if word in e_params.keys():
            if sentiment in e_params[word].keys():
                e_params[word][sentiment] += 1
            else:
                e_params[word][sentiment] = 1
        else:
            new_word = {}
            new_word[sentiment] = 1
            e_params[word] = new_word

        # update t_params
        if sentiment in t_params.keys():
            if prev in t_params[sentiment].keys():
                t_params[sentiment][prev] += 1
            else:
                t_params[sentiment][prev] = 1
        else:
            new_sentiment = {}
            new_sentiment[prev] = 1
            t_params[sentiment] = new_sentiment
        prev = sentiment
    
    sentiment_list = t_params[end]
    if prev in sentiment_list.keys():
        sentiment_list[prev] +=1
    else:
        sentiment_list[prev] =1
    prev = 'START'
    
    # predicted
    for pentry in predicted:
        pair = pentry.split()
        word = pair[0]
        sentiment = pair[1]

        # update e_params
        if word in e_params.keys():
            if sentiment in e_params[word].keys():
                e_params[word][sentiment] -= 1
            else:
                e_params[word][sentiment] = -1
        else:
            new_word = {}
            new_word[sentiment] = -1
            e_params[word] = new_word

        # update t_params
        if sentiment in t_params.keys():
            if prev in t_params[sentiment].keys():
                t_params[sentiment][prev] -= 1
            else:
                t_params[sentiment][prev] =-1
        else:
            new_sentiment = {}
            new_sentiment[prev] = -1
            t_params[sentiment] = new_sentiment
        prev = sentiment

    sentiment_list = t_params[end]
    if prev in sentiment_list.keys():
        sentiment_list[prev] -=1
    else:
        sentiment_list[prev] =-1
            
    return (e_params, t_params)

def write_params_to_file(iteration, e_params, t_params):
    pickle.dump(e_params, open( "e_para_"+str(iteration)+".p", "wb" ))
    pickle.dump(t_params, open( "t_para_"+str(iteration)+".p", "wb" ))
    

def predict(input_file, input_pre, output_file, transition_params, emission_params, iteration):
    if iteration == 0:
        iteration = ""
    else:
        iteration = str(iteration)
    labels = ['B-neutral', 'I-neutral', 'B-negative', 'I-negative', 'B-positive', 'I-positive', 'O']
    sentences = []
    original_sentences = []
    with open(input_file, encoding ='utf-8') as ifile, open(input_pre, encoding = 'utf-8') as pfile, codecs.open(output_file+iteration, 'w', 'utf-8-sig') as ofile:
        sentence = []
        original_sentence = []
        
        for iline, pline in zip(ifile, pfile):
            if len(iline.split())!=0:
                original_sentence.append(iline.split()[0])
                sentence.append(pline.split()[0])
            else:
                sentences.append(sentence)
                original_sentences.append(original_sentence)
                sentence = []
                original_sentence = []
        for i in range(len(sentences)):
            nodes = calculate_node_scores(sentences[i], transition_params, emission_params, labels)
            predicted_sentence = backtracking(original_sentences[i],nodes)
            for word in predicted_sentence:
                ofile.write(word+'\n')
            ofile.write("\n")


def out(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return result.stdout

def plot_graph(scores,iteration):
    x=[]
    y=[]
    for s in scores:
        x.append(s[0])
        y.append(s[1])
    plt.plot(x,y)
    plt.axis([0, iteration, 0, 1])
    plt.show()


def main(train_file, train_pre, output_file, iterations):
    # preprocess train file
    include_sentiment = True
    preprocessing(train_file, train_pre, include_sentiment)
    # initialize params to 0
    e_params, t_params = initialize_params(train_pre)
    scores = []
    for i in range(1, iterations+1):
        print("Iteration: "+str(i))
        # update params using perceptron algorithm
        e_params, t_params = perceptron_algorithm(train_pre, e_params, t_params)
        # save params into file
        write_params_to_file(i, e_params, t_params)
        # use params to predict on train file
        predict(train_file,train_pre, output_file, e_params, t_params, i)
        # calculate fscore & append to scores list
        output = out(["python", "evalResult.py","train", "train_out_"+str(i)])
        output2 = output.split('\n')
        if len(output2) > 5:
            fscore = output2[12]
            scores.append([i,float(fscore[14:])])
        else:
            scores.append([i,0])
    # sort scores
    sortedscores = sorted(scores, key=lambda x: x[1], reverse=True)
    print ("Best iteration, score: "+ str(sortedscores[0]))
    # plot graph
    plot_graph(scores, iterations)
    return sortedscores[0][0]

def run_test(input_file, input_pre, output_file, best_iter, include_sentiment):
    preprocessing(input_file, input_pre, include_sentiment)
    e_params = pickle.load( open( "e_para_"+str(best_iter)+".p", "rb" ))
    t_params = pickle.load( open( "t_para_"+str(best_iter)+".p", "rb" ))
    predict(input_file, input_pre, output_file, e_params, t_params, 0)

import sys

iterations = int(sys.argv[1])

dev_input = './dev.out'
dev_pre = './dev_pre'
dev_out = './dev.p5.out'

test_input = './test.in'
test_pre = './test_pre'
test_out = './test.p5.out'


# run program to find best iteration
print("Begin training of model")
best_iter = main(train, train_pre, output_file, iterations)

# predict on dev.in to produce dev.p5.out
print("Predicting on dev.in")
include_sentiment = True
run_test(dev_input, dev_pre, dev_out, best_iter, include_sentiment)

# predict on test.in to produce test.p5.out
print("Predicting on test.in")
include_sentiment = False
run_test(test_input, test_pre, test_out, best_iter, include_sentiment)

print("Ending of Program")
sys.exit()

