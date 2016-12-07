
# coding: utf-8

# In[ ]:

import codecs
train_file = "./train"
dev_in = "./dev.in"
topk_out = "./dev.p4.out"


def emission_params(train_file):
    with open(train_file, encoding = 'utf-8') as file:
        emission_count= {}
        label_count={}
        for line in file:
            pair = line.split()
            if len(line.split())!=0:
                #add 1 to count of (Xi, Yi)
                word = pair[0]
                sentiment = pair[1]
                if word in emission_count.keys():
                    if sentiment in emission_count[word].keys():
                        emission_count[word][sentiment] +=1
                    else:
                        sentiments = emission_count[word]
                        sentiments[sentiment] = 1
                else:
                    sentiment_count = {}
                    sentiment_count[sentiment] = 1
                    emission_count[word]=sentiment_count
    
                #add 1 to count of label Yi
                if sentiment in label_count.keys():
                    label_count[sentiment]+=1
                else:
                    label_count[sentiment]=1
        for keya in emission_count.keys():
            for keyb in emission_count[keya].keys():
                emission_count[keya][keyb]/=(label_count[keyb]+1)
        new_word = {}
        for key in label_count.keys():
            new_word[key] = 1/(label_count[key]+1)
        emission_count['new_word'] = new_word
       
        return (emission_count,label_count)
    
def transition_params(train_file):
    transition_count= {}
    state_count={}
    prev = 'START'
    end = 'STOP'
    state_count[prev] = 0
    state_count[end] = 0
    transition_count[end] = {}
    with open(train_file, encoding = 'utf-8') as file:    
        for line in file:
            pair = line.split()
            if len(pair)!= 0:
                sentiment = pair[1]
                # add prev to sentiment transition count
                if sentiment in transition_count.keys():
                    sentiment_list = transition_count[sentiment]
                    if prev in sentiment_list.keys():
                        sentiment_list[prev] += 1
                    else:
                        sentiment_list[prev] = 1
                else:
                    new_sentiment = {}
                    new_sentiment[prev] = 1
                    transition_count[sentiment] = new_sentiment

                # add to start and stop state counts
                if prev == 'START':
                    state_count[prev] += 1
                    state_count[end] += 1

                # add to state count  
                if sentiment in state_count.keys():
                    state_count[sentiment]+=1
                else:
                    state_count[sentiment]=1
              
                prev = sentiment

            else:
                sentiment_list = transition_count[end]
                if prev in sentiment_list.keys():
                    sentiment_list[prev] +=1
                else:
                    sentiment_list[prev] =1   
                prev = 'START'
    for V in transition_count.keys():
        for U in transition_count[V].keys():
            transition_count[V][U] /= state_count[U]
    return transition_count

def viterbi_algo_topk(test_file, output_file, transition_params, emission_params, labels, top_k, i_th):
    sentences = []

    with open(test_file, encoding ='utf-8') as ifile, codecs.open(output_file, 'w', 'utf-8-sig') as ofile:
        sentence = []
        for line in ifile:
            if len(line.split())!=0:
                sentence.append(line.split()[0])
            else:
                sentences.append(sentence)
                sentence = []
        
        for s in sentences:
            nodes = calculate_topk_node_scores(s,transition_params, emission_params, labels, top_k)
            labelled_sentence = backtracking_topk(s,nodes, i_th)
            for word in labelled_sentence:
                ofile.write(word+'\n')
            ofile.write("\n")


def calculate_topk_node_scores(s, transition_params, emission_params, labels, top_k):
    nodes = {}
    #base case
    nodes[0] = {'START':[[1,'nil',0]]}
    #recursive
    for k in range (1, len(s)+1): #for each word
        X = s[k-1]
        for V in labels.keys(): #for each node
            prev_nodes_dict = nodes[k-1] #access prev nodes
            #emission params
            if X in emission_params.keys():
                emission_labels = emission_params[X]

                if V in emission_labels:
                    b = emission_labels[V]
                else:
                    b = 0
            else:
                b = emission_params['new_word'][V]  
            scores = []
            for U in prev_nodes_dict.keys():
                #transitionparams
                prev_states = transition_params[V]
                if U in prev_states.keys():
                    a = prev_states[U]
                else:
                    a = 0
                index = 0
                for prev_k_nodes in prev_nodes_dict[U]:
                    #prev node score
                    score = prev_k_nodes[0]*a*b
                    scores.append([score, U, index])
                    index += 1
            
            #take top k scores
            scores.sort(key=lambda x: x[0],reverse=True)
            topk_scores = scores[:top_k]
            if k in nodes.keys():
                nodes[k][V] = topk_scores
            else:
                new_dict = {V:topk_scores}
                nodes[k] = new_dict
            
    #end case
    prev_nodes_dict = nodes[len(s)]
    scores = []
    for U in prev_nodes_dict.keys():
        #transition
        prev_states = transition_params['STOP']
        if U in prev_states.keys():
            a = prev_states[U]
        else:
            a = 0
        #prev node score
        index = 0
        for prev_k_nodes in prev_nodes_dict[U]:
            score = prev_k_nodes[0]*a
            scores.append([score, U, index])
            index += 1
    scores.sort(key=lambda x: x[0], reverse=True)
    topk_scores = scores[:top_k]
    indiv_node = {'STOP': topk_scores}
    nodes[len(s)+1]=indiv_node
    
    return nodes


def backtracking_topk(s, nodes, i_th):
    prev_state = 'STOP'
    prev_index = 0
    for i in range(len(s)+1, 1,-1):
        if i==len(s)+1:
            prev_node = nodes[i][prev_state][i_th-1]
        else:
            prev_node = nodes[i][prev_state][prev_index]
        prev_state = prev_node[1]
        prev_index = prev_node[2]
        s[i-2] += " "+prev_state
    return s

import sys
top_k = int(sys.argv[1])
i_th = int(sys.argv[2])
suffix = ""

if i_th == 1:
    suffix = "st"
elif i_th == 2:
    suffix ="nd"
elif i_th == 3:
    suffix = "rd"
else:
    suffix = "th"
    
print("Start program")
# obtain e_params
print("obtaining emission params")
e_params, label_count = emission_params(train_file)
# obtain t_params
print("obtaining transition params")
t_params = transition_params(train_file)
# perform viterbi to find top k scores and output ith best output
print("finding top "+str(top_k)+" best output sequences. Outputting "+str(i_th)+suffix+" best sequence.")
viterbi_algo_topk(dev_in, topk_out, t_params, e_params, label_count, top_k, i_th)


# In[ ]:



