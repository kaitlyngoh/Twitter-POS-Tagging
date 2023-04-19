
# Implement the six functions below
import pandas as pd
import contractions
import re



def read_file(filename):
    text_file = open(filename, "r")
    text_file = text_file.readlines()
    return text_file


def helperFnQn2_3():
    corpus = read_file("twitter_train.txt")
    corpus = list(filter(lambda x: x != "\n", corpus))
    token_tag_list = []
    for element in corpus:
        token = element.split("\t")[0]

        if 'http' not in token:
            token = token.lower()
        # if token.startswith("@user"):
        #     token = "@user"
        # if token.startswith("@http"):
        #     token = "http"

        tag = element.split("\t")[1].split("\n")[0]
        lst = (token,tag)
        token_tag_list.append(lst)
    unique_tokens = set()
    dict_of_tag ={}
    for token,tag in token_tag_list:
        unique_tokens.add(token)
        if tag not in dict_of_tag:
            dict_of_tag[tag] = {"tag_appeared_occurrence": 1, "list of words": {token: 1}}
        else:
            dict_of_tag[tag]["tag_appeared_occurrence"] += 1
            if token not in dict_of_tag[tag]["list of words"]:
                dict_of_tag[tag]["list of words"][token] = 1
            else:
                dict_of_tag[tag]["list of words"][token] += 1
    delta = 0.1
    num_words = len(unique_tokens)
    for tag in dict_of_tag:
        yj = dict_of_tag[tag]["tag_appeared_occurrence"]  # how many times tag appear
        dict_of_tag[tag]["list of words"]["WordNotFound"] = 0  #delta replaces the number of times the word appears
        for word in dict_of_tag[tag]["list of words"]:
            yjxw = dict_of_tag[tag]["list of words"][word]
            bjw = (yjxw + delta) / (yj + (delta * (num_words + 1)))
            dict_of_tag[tag]["list of words"][word] = bjw
    tag_occurrence_dict = []
    for tag in dict_of_tag:
        count = dict_of_tag[tag]["tag_appeared_occurrence"]
        tag_occurrence_dict.append((tag,count))
    df_tag_count = pd.DataFrame(tag_occurrence_dict, columns=["tag", "count"])
    fileVariable = open('tag_count.txt', 'r+')
    fileVariable.truncate(0)
    fileVariable.close()
    df_tag_count.to_csv("tag_count.txt", index=None, sep='\t', mode='a')
    lst_token_tag_bjw =[]
    for tag in dict_of_tag:
        #get token, bjw
        lst_words = dict_of_tag[tag]["list of words"]
        for word in lst_words:
            bjw = dict_of_tag[tag]["list of words"][word]
            lst_token_tag_bjw.append((word,tag,bjw))
    df = pd.DataFrame(lst_token_tag_bjw,columns=["token","tag","bjw"])

    fileVariable = open('output_probs.txt', 'r+')
    fileVariable.truncate(0)
    fileVariable.close()
    df.to_csv("output_probs.txt", index=None, sep='\t', mode='a')

    fileVariable = open('naive_output_probs.txt', 'r+')
    fileVariable.truncate(0)
    fileVariable.close()
    df.to_csv("naive_output_probs.txt", index=None, sep='\t', mode='a')

# helperFnQn2_3()


def helperFnQn4_bjw():
    corpus = read_file("twitter_train.txt")
    corpus = list(filter(lambda x: x != "\n", corpus))
    token_tag_list = []
    for element in corpus:
        token = element.split("\t")[0]

        if 'http' not in token:
            token = token.lower()
        # if token.startswith("@user"):
        #     token = "@user"
        # if token.startswith("@http"):
        #     token = "http"

        tag = element.split("\t")[1].split("\n")[0]
        lst = (token,tag)
        token_tag_list.append(lst)
    unique_tokens = set()
    dict_of_tag ={}
    for token,tag in token_tag_list:
        unique_tokens.add(token)
        if tag not in dict_of_tag:
            dict_of_tag[tag] = {"tag_appeared_occurrence": 1, "list of words": {token: 1}}
        else:
            dict_of_tag[tag]["tag_appeared_occurrence"] += 1
            if token not in dict_of_tag[tag]["list of words"]:
                dict_of_tag[tag]["list of words"][token] = 1
            else:
                dict_of_tag[tag]["list of words"][token] += 1
    delta = 0.1
    num_words = len(unique_tokens)
    for tag in dict_of_tag:
        yj = dict_of_tag[tag]["tag_appeared_occurrence"]  # how many times tag appear
        dict_of_tag[tag]["list of words"]["WordNotFound"] = 0  #delta replaces the number of times the word appears
        for word in dict_of_tag[tag]["list of words"]:
            yjxw = dict_of_tag[tag]["list of words"][word]
            bjw = (yjxw + delta) / (yj + (delta * (num_words + 1)))
            dict_of_tag[tag]["list of words"][word] = bjw
    tag_occurrence_dict = []
    for tag in dict_of_tag:
        count = dict_of_tag[tag]["tag_appeared_occurrence"]
        tag_occurrence_dict.append((tag,count))
    df_tag_count = pd.DataFrame(tag_occurrence_dict, columns=["tag", "count"])
    fileVariable = open('tag_count.txt', 'r+')
    fileVariable.truncate(0)
    fileVariable.close()
    df_tag_count.to_csv("tag_count.txt", index=None, sep='\t', mode='a')
    lst_token_tag_bjw =[]
    for tag in dict_of_tag:
        #get token, bjw
        lst_words = dict_of_tag[tag]["list of words"]
        for word in lst_words:
            bjw = dict_of_tag[tag]["list of words"][word]
            lst_token_tag_bjw.append((word,tag,bjw))
    df = pd.DataFrame(lst_token_tag_bjw,columns=["token","tag","bjw"])

    fileVariable = open('output_probs.txt', 'r+')
    fileVariable.truncate(0)
    fileVariable.close()
    df.to_csv("output_probs.txt", index=None, sep='\t', mode='a')

helperFnQn4_bjw()

def helperFnQn4_aij():
    delta = 0.1
    corpus = read_file("twitter_train.txt")
    idx = [0]
    for i in range(len(corpus)):
        #find the index of @
        element = corpus[i]
        if element == "\n":
            idx.append(i)
    corpus_user = []
    for i in range(len(idx)):
        num = idx[i]
        if (i == len(idx) - 1):
            tweet = corpus[num:]
        else:
            nextNum = idx[i + 1]
            tweet = corpus[num:nextNum]
        tweet.insert(0, "hmmstart\tSTART\n")
        tweet.append("hmmend\tSTOP\n")
        corpus_user.append(tweet)
    corpus_user
    train_data = []
    for lst in corpus_user:
        lst = list(filter(lambda x: x != "\n", lst))
        tweet = []
        for ele in lst:
            token_tag_pair = []
            token = ele.split("\t")[0]
            tag = ele.split("\t")[1].split("\n")[0]

            # if 'http' not in token:
            #     token = token.lower()
            # if token.startswith("@user"):
            #     token = "@user"
            # if token.startswith("@http"):
            #     token = "http"

            token_tag_pair = (token, tag)
            tweet.append(token_tag_pair)
        train_data.append(tweet)
    train_data
    dic_stop = {}
    # From  | To
    # A
    # i -> j
    transitionListPairs = []
    len_seq = len(train_data)
    for tweet in train_data:
        for i in range(len(tweet)):
            if i == (len(tweet) - 1):
                stop_state = tweet[i][1]
                if stop_state == "START":
                    continue
                if stop_state not in dic_stop:
                    dic_stop[stop_state] = 1
                else:
                    dic_stop[stop_state] += 1
            else:
                from_tag = tweet[i][1]
                to_tag = tweet[i+1][1]
                transitionListPairs.append([from_tag,to_tag])
    transitionListPairs
    transitionPairs = pd.DataFrame(transitionListPairs, columns = ["i", "j"])
    ### Loop thru each i -> j transition create a key:value
    ### key is i->j
    ### value is count
    dic_ij_count = {}
    for index, row in transitionPairs.iterrows():
        i = row.i
        j = row.j
        pair = (i,j)
        if pair not in dic_ij_count:
            dic_ij_count[pair] = 1
        else:
            dic_ij_count[pair] = dic_ij_count[pair] + 1
    dic_ij_prob = {}
    ###create transition probability dictionary
    df_tag_count = pd.read_csv("tag_count.txt", sep="\t")
    uniqueTags = list(df_tag_count.tag)
    start_count = len(train_data)
    start_row = {'tag': 'START', 'count': start_count}
    df_tag_count.loc[len(df_tag_count)] = start_row

    for key,value in dic_ij_count.items():
        i = key[0]
        count_i = df_tag_count.loc[df_tag_count['tag'] == i]["count"].values[0]
        p = value/count_i
        dic_ij_prob[key] = p
    # for key,value in dic_stop.items():
    #     i = key
    #     count = value
    #     pair = (i, "STOP")
    #     probability = count/len_seq
    #     dic_ij_prob[pair] = probability
    #Generated all the possible i->j pairs that we see in the train data
    dic_ij_prob
    #Generate all the possible unseen pairs
    all_possible_start = uniqueTags.copy()
    all_possible_start.append("START")

    all_possible_end = uniqueTags.copy()
    all_possible_end.append("STOP")

    df_from_to = pd.DataFrame(list(dic_ij_prob.items()), columns=["i,j", "prob"])
    df_count = pd.DataFrame()
    df_count[['i', 'j']] = df_from_to['i,j'].apply(lambda x: pd.Series(x))

    for i in all_possible_start:
        for j in all_possible_end:
            pair = (i,j)
            if pair in dic_ij_prob:
                pass
            else:
                #create the pair
                count_i = len(df_count[df_count['i'] == i])
                # count_i = df_tag_count.loc[df_tag_count['tag'] == i]["count"].values[0]
                numerator = delta + 0
                denominator = count_i + (delta * (len(uniqueTags) + 1))
                probaility = numerator/denominator
                dic_ij_prob[pair] = probaility
                # print(probaility)

    df_ij_prob = pd.DataFrame(list(dic_ij_prob.items()), columns = ["i,j", "prob"])
    df_i_j_prob = pd.DataFrame()
    df_i_j_prob[['i', 'j']] = df_ij_prob['i,j'].apply(lambda x: pd.Series(x))
    df_i_j_prob['prob'] = df_ij_prob["prob"]

    fileVariable = open('trans_probs.txt', 'r+')
    fileVariable.truncate(0)
    fileVariable.close()
    df_i_j_prob.to_csv("trans_probs.txt", index=None, sep='\t', mode='a')

helperFnQn4_aij()

import numpy as np

def traverseBp(idx,bpMatrix,possible_tags,sequence_length):
    tag_result = []
    sequence_length
    idxFrom = idx
    for i in range(sequence_length-1,-1,-1):
        tag = possible_tags[idxFrom]
        idxFrom = int(bpMatrix[i][idxFrom])
        tag_result.append(tag)
    tag_result.reverse()
    return tag_result

def initialize(tweet, possible_tags, aij_dict, bjw_dict):
    # Sequence 1: firstword -> x
    # create Pi(1,x) <- Generate from possible tags
    # create backpointer <- Init as 0

    dic_aij = aij_dict["START"]
    # key is "possible start tag" value is probability

    sequence_length = len(tweet)
    number_of_state_to_transition_to = len(possible_tags)

    viterbiMatrix = np.zeros((sequence_length, number_of_state_to_transition_to))
    backpointerMatrix = np.zeros((sequence_length, number_of_state_to_transition_to))

    firstWord = tweet[0]

    for i in range(len(possible_tags)):
        tag = possible_tags[i]
        # transition probability
        aij = dic_aij[tag]

        #output probability
        if bjw_dict.get(firstWord) == None:
            #word does not exist
            firstWord = "WordNotFound"
            bjw = bjw_dict[firstWord][tag]
        else:
            if bjw_dict[firstWord].get(tag) == None:
                word = "WordNotFound"
                bjw = bjw_dict[word][tag]
            else:
                bjw = bjw_dict[firstWord][tag]

        probability = aij * bjw
        viterbiMatrix[0][i] = probability
    return viterbiMatrix, backpointerMatrix


def viterbiFn(tweet, possible_tags, aij_dict, bjw_dict):
    # Initialization Step
    #return 2 matrix
    vMatrix, bpMatrix = initialize(tweet, possible_tags, aij_dict, bjw_dict)

    # Iteration Step
    sequence_length = len(tweet)
    numStates = len(possible_tags)

    # for loop from row = 1 to sequence length
    # inner loop is length of states
    for word_index in range(1, sequence_length):
        for state_index in range(numStates):
            word = tweet[word_index]
            jState = possible_tags[state_index]
            #check if word exists
            if bjw_dict.get(word) == None:
                # word does not exist
                word = "WordNotFound"
                bjw = bjw_dict[word][jState]
            else:
                if bjw_dict[word].get(jState) == None:
                    word = "WordNotFound"
                    bjw = bjw_dict[word][jState]
                else:
                    bjw = bjw_dict[word][jState]

            lst_get_max = []
            # get highest probability and the corresponding index from lst_get_max
            for iState in range(len(possible_tags)):
                prev_prob = vMatrix[word_index-1][iState]
                aij = aij_dict[possible_tags[iState]][jState]
                p = prev_prob * aij * bjw
                lst_get_max.append(p)
                # print(possible_tags[iState], jState)
            idx_max = lst_get_max.index(max(lst_get_max))
            max_prob = max(lst_get_max)
            vMatrix[word_index][state_index] = max_prob
            bpMatrix[word_index][state_index] = idx_max

    ### END STATE
    lst_get_max_stop = []
    for state_index in range(numStates):
        istate = possible_tags[state_index]
        #get a istate -> endstate
        a_i_stop = aij_dict[istate]["STOP"]
        prev_prob = vMatrix[sequence_length-1][state_index]
        final_prob = prev_prob * a_i_stop
        lst_get_max_stop.append(final_prob)
    idx_max_final = lst_get_max_stop.index(max(lst_get_max_stop))
    max_prob_final = max(lst_get_max_stop)
    result_tags = traverseBp(idx_max_final,bpMatrix,possible_tags,sequence_length)
    return result_tags



def helperFnQn5_bjw():
    corpus = read_file("twitter_train.txt")
    corpus = list(filter(lambda x: x != "\n", corpus))
    token_tag_list = []
    for element in corpus:
        token = element.split("\t")[0]
        web_pattern = r'\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
        websites = re.findall(web_pattern, token)
        if websites:
            #this word is a website
            token = "hmm_website"
        else:
            token = token.lower()
            user_pattern = r'@user\w+'
            users = re.findall(user_pattern, token)
            if users:
                # this word is a user
                token = "hmm_user"
        if token.startswith("#"):
            token = "hmm_#"

        digits_pattern = r'\b\d+(?:\.\d+)?\b|\$\d+(?:,\d+)*(?:\.\d+)?|\d{1,2}:\d{2}(?::\d{2})?'
        digits = re.findall(digits_pattern, token)
        if digits:
            token = "hmm_num"
        contract_words = contractions.fix(token)
        if ((contract_words != token) and len(contract_words.split(" ")) == 1):
            # print("\n")
            # print(token)
            # print(contract_words)
            # print("\n")
            token = contract_words
        # token = spell(token)

        tag = element.split("\t")[1].split("\n")[0]
        lst = (token,tag)
        token_tag_list.append(lst)
    unique_tokens = set()
    dict_of_tag ={}
    for token,tag in token_tag_list:
        unique_tokens.add(token)
        if tag not in dict_of_tag:
            dict_of_tag[tag] = {"tag_appeared_occurrence": 1, "list of words": {token: 1}}
        else:
            dict_of_tag[tag]["tag_appeared_occurrence"] += 1
            if token not in dict_of_tag[tag]["list of words"]:
                dict_of_tag[tag]["list of words"][token] = 1
            else:
                dict_of_tag[tag]["list of words"][token] += 1
    delta = 0.01
    num_words = len(unique_tokens)
    for tag in dict_of_tag:
        yj = dict_of_tag[tag]["tag_appeared_occurrence"]  # how many times tag appear
        dict_of_tag[tag]["list of words"]["WordNotFound"] = 0  #delta replaces the number of times the word appears
        for word in dict_of_tag[tag]["list of words"]:
            yjxw = dict_of_tag[tag]["list of words"][word]
            bjw = (yjxw + delta) / (yj + (delta * (num_words + 1)))
            dict_of_tag[tag]["list of words"][word] = bjw
    tag_occurrence_dict = []
    for tag in dict_of_tag:
        count = dict_of_tag[tag]["tag_appeared_occurrence"]
        tag_occurrence_dict.append((tag,count))
    df_tag_count = pd.DataFrame(tag_occurrence_dict, columns=["tag", "count"])
    fileVariable = open('tag_count.txt', 'r+')
    fileVariable.truncate(0)
    fileVariable.close()
    df_tag_count.to_csv("tag_count.txt", index=None, sep='\t', mode='a')
    lst_token_tag_bjw =[]
    for tag in dict_of_tag:
        #get token, bjw
        lst_words = dict_of_tag[tag]["list of words"]
        for word in lst_words:
            bjw = dict_of_tag[tag]["list of words"][word]
            lst_token_tag_bjw.append((word,tag,bjw))
    df = pd.DataFrame(lst_token_tag_bjw,columns=["token","tag","bjw"])

    fileVariable = open('output_probs2.txt', 'r+')
    fileVariable.truncate(0)
    fileVariable.close()
    df.to_csv("output_probs2.txt", index=None, sep='\t', mode='a')

helperFnQn5_bjw()

def helperFnQn5_aij():
    delta = 0.01
    corpus = read_file("twitter_train.txt")
    idx = [0]
    for i in range(len(corpus)):
        #find the index of @
        element = corpus[i]
        if element == "\n":
            idx.append(i)
    corpus_user = []
    for i in range(len(idx)):
        num = idx[i]
        if (i == len(idx) - 1):
            tweet = corpus[num:]
        else:
            nextNum = idx[i + 1]
            tweet = corpus[num:nextNum]
        tweet.insert(0, "hmmstart\tSTART\n")
        tweet.append("hmmend\tSTOP\n")
        corpus_user.append(tweet)
    corpus_user
    train_data = []
    n = 0
    for lst in corpus_user:
        lst = list(filter(lambda x: x != "\n", lst))
        tweet = []
        for ele in lst:
            token_tag_pair = []
            token = ele.split("\t")[0]
            tag = ele.split("\t")[1].split("\n")[0]

            web_pattern = r'\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
            websites = re.findall(web_pattern, token)
            if websites:
                # this word is a website
                token = "hmm_website"
            else:
                token = token.lower()
                user_pattern = r'@user\w+'
                users = re.findall(user_pattern, token)
                if users:
                    # this word is a user
                    token = "hmm_user"
            if token.startswith("#"):
                token = "hmm_#"
            digits_pattern = r'\b\d+(?:\.\d+)?\b|\$\d+(?:,\d+)*(?:\.\d+)?|\d{1,2}:\d{2}(?::\d{2})?'
            digits = re.findall(digits_pattern, token)
            if digits:
                token = "hmm_num"
            contract_words = contractions.fix(token)
            if ((contract_words != token) and len(contract_words.split(" ")) == 1):
                # print("\n")
                # print(token)
                # print(contract_words)
                # print("\n")
                token = contract_words
            # token = spell(token)
            token_tag_pair = (token, tag)
            tweet.append(token_tag_pair)
        train_data.append(tweet)
    train_data
    dic_stop = {}
    # From  | To
    # A
    # i -> j
    transitionListPairs = []
    len_seq = len(train_data)
    for tweet in train_data:
        for i in range(len(tweet)):
            if i == (len(tweet) - 1):
                stop_state = tweet[i][1]
                if stop_state == "START":
                    continue
                if stop_state not in dic_stop:
                    dic_stop[stop_state] = 1
                else:
                    dic_stop[stop_state] += 1
            else:
                from_tag = tweet[i][1]
                to_tag = tweet[i+1][1]
                transitionListPairs.append([from_tag,to_tag])
    transitionListPairs
    transitionPairs = pd.DataFrame(transitionListPairs, columns = ["i", "j"])
    ### Loop thru each i -> j transition create a key:value
    ### key is i->j
    ### value is count
    dic_ij_count = {}
    for index, row in transitionPairs.iterrows():
        i = row.i
        j = row.j
        pair = (i,j)
        if pair not in dic_ij_count:
            dic_ij_count[pair] = 1
        else:
            dic_ij_count[pair] = dic_ij_count[pair] + 1
    dic_ij_prob = {}
    ###create transition probability dictionary
    df_tag_count = pd.read_csv("tag_count.txt", sep="\t")
    uniqueTags = list(df_tag_count.tag)
    start_count = len(train_data)
    start_row = {'tag': 'START', 'count': start_count}
    df_tag_count.loc[len(df_tag_count)] = start_row

    for key,value in dic_ij_count.items():
        i = key[0]
        count_i = df_tag_count.loc[df_tag_count['tag'] == i]["count"].values[0]
        p = value/count_i
        dic_ij_prob[key] = p
    # for key,value in dic_stop.items():
    #     i = key
    #     count = value
    #     pair = (i, "STOP")
    #     probability = count/len_seq
    #     dic_ij_prob[pair] = probability
    #Generated all the possible i->j pairs that we see in the train data
    dic_ij_prob
    #Generate all the possible unseen pairs
    all_possible_start = uniqueTags.copy()
    all_possible_start.append("START")

    all_possible_end = uniqueTags.copy()
    all_possible_end.append("STOP")

    df_from_to = pd.DataFrame(list(dic_ij_prob.items()), columns=["i,j", "prob"])
    df_count = pd.DataFrame()
    df_count[['i', 'j']] = df_from_to['i,j'].apply(lambda x: pd.Series(x))

    for i in all_possible_start:
        for j in all_possible_end:
            pair = (i,j)
            if pair in dic_ij_prob:
                pass
            else:
                #create the pair
                count_i = len(df_count[df_count['i'] == i])
                # count_i = df_tag_count.loc[df_tag_count['tag'] == i]["count"].values[0]
                numerator = delta + 0
                denominator = count_i + (delta * (len(uniqueTags) + 1))
                probaility = numerator/denominator
                dic_ij_prob[pair] = probaility
                # print(probaility)

    df_ij_prob = pd.DataFrame(list(dic_ij_prob.items()), columns = ["i,j", "prob"])
    df_i_j_prob = pd.DataFrame()
    df_i_j_prob[['i', 'j']] = df_ij_prob['i,j'].apply(lambda x: pd.Series(x))
    df_i_j_prob['prob'] = df_ij_prob["prob"]

    fileVariable = open('trans_probs2.txt', 'r+')
    fileVariable.truncate(0)
    fileVariable.close()
    df_i_j_prob.to_csv("trans_probs2.txt", index=None, sep='\t', mode='a')

helperFnQn5_aij()






def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    '''Naive prediction accuracy:     945/1378 = 0.6857764876632801'''
    qn2a = read_file(in_output_probs_filename)
    to_predict = read_file(in_test_filename)
    to_predict = list(filter(lambda x: x != "\n", to_predict))
    df_2a = pd.read_csv(in_output_probs_filename, sep="\t")
    df_out_prediction = pd.DataFrame()
    lst_out_prediction = []
    for element in to_predict:
        element = element.split("\n")[0].lower()
        # if element.startswith("@user"):
        #     element = "@user"
        # if element.startswith('http'):
        #     element = "http"
        df_repeated_rows = df_2a.loc[df_2a['token'] == element]
        if df_repeated_rows.empty:
            element = "WordNotFound"
            df_repeated_rows = df_2a.loc[df_2a['token'] == element]
        max_index = df_repeated_rows.bjw.idxmax()
        token = df_2a.loc[max_index, : ]['token']
        tag = df_2a.loc[max_index, : ]['tag']
        lst_out_prediction.append(tag)
    fileVariable = open(out_prediction_filename, 'r+')
    fileVariable.truncate(0)
    fileVariable.close()
    with open(out_prediction_filename, 'w') as file:
        for item in lst_out_prediction:
            file.write(f"{item}\n")

def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    '''Naive prediction2 accuracy:    984/1378 = 0.714078374455733'''

    '''
    3(a) How do you compute the right-hand side of this equation?
    P(y=j | x = w) =  P(x = w| y = j ) * P(y = j) / P(x = w)
    P(x=w) represent the probability of observing the word w in the training data.
    Since the training data does not change,
    P(x=w) remains the same for every candidate, hence we can ignore this term.
    '''
    qn3b = read_file(in_output_probs_filename)
    to_predict = read_file(in_test_filename)
    to_predict = list(filter(lambda x: x != "\n", to_predict))
    df_3b = pd.read_csv(in_output_probs_filename, sep="\t")
    df_tag_count = pd.read_csv("tag_count.txt", sep="\t")
    token_tag_bjw_qn3 = []
    total_tag_occurrence = df_tag_count['count'].sum()
    for index, row in df_3b.iterrows():
        token = row.token
        tag = row.tag
        bjw = row.bjw
        num_occurrence_tag = df_tag_count.loc[df_tag_count['tag'] == tag]["count"].values[0]
        pyj = num_occurrence_tag/total_tag_occurrence
        newbjw = pyj * bjw
        token_tag_bjw_qn3.append((token,tag,newbjw))
    df_newBjw = pd.DataFrame(token_tag_bjw_qn3, columns=["token", "tag", "bjw"])

    lst_out_prediction = []
    for element in to_predict:
        element = element.split("\n")[0].lower()
        # if element.startswith("@user"):
        #     element = "@user"
        # if element.startswith('http'):
        #     element = "http"
        df_repeated_rows = df_newBjw.loc[df_newBjw['token'] == element]
        if df_repeated_rows.empty:
            element = "WordNotFound"
            df_repeated_rows = df_newBjw.loc[df_newBjw['token'] == element]
        max_index = df_repeated_rows.bjw.idxmax()
        token = df_newBjw.loc[max_index, : ]['token']
        tag = df_newBjw.loc[max_index, : ]['tag']
        lst_out_prediction.append(tag)
    fileVariable = open(out_prediction_filename, 'r+')
    fileVariable.truncate(0)
    fileVariable.close()
    with open(out_prediction_filename, 'w') as file:
        for item in lst_out_prediction:
            file.write(f"{item}\n")

def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    ''' Viterbi prediction accuracy:   1069/1378 = 0.7757619738751814 '''
    # read the transition probability file and output probability file and store into a dataframe
    trans_prob = pd.read_csv(in_trans_probs_filename, sep="\t")

    bjw = pd.read_csv(in_output_probs_filename, sep="\t")

    #unpack trans probability dict for ease of accessing value
    trans_prob_dic = {}
    for _, row in trans_prob.iterrows():
        if row['i'] not in trans_prob_dic:
            trans_prob_dic[row['i']] = {}
        trans_prob_dic[row['i']][row['j']] = row['prob']
    #unpack bjw probability dict for ease of accessing value
    bjw_dict = {}
    for _, row in bjw.iterrows():
        if row['token'] not in bjw_dict:
            bjw_dict[row['token']] = {}
        bjw_dict[row['token']][row['tag']] = row['bjw']

    #reading the test data file and split the tweets by "\n"
    #1) identify the index of the "/n"
    #2) extract the tweet using the index of "/n"
    corpus = read_file(in_test_filename)
    idx = [-1]
    for i in range(len(corpus)):
        element = corpus[i]
        if element == "\n":
            idx.append(i)
    corpus_user = []
    for i in range(len(idx)):
        num = idx[i]
        if (i == len(idx) - 1):
            tweet = corpus[num:]
        else:
            nextNum = idx[i + 1]
            tweet = corpus[num+1:nextNum]
        corpus_user.append(tweet)
    #list of all the tweets stored in corpus user
    corpus_user

    test_data = []
    #slight preprocessing to lowercase all the token except for websites
    for lst in corpus_user:
        if len(lst) == 1:
            if lst[0] == "\n":
                continue
        tweet = []
        for ele in lst:
            token = ele.split("\t")[0].split("\n")[0]
            if 'http' not in token:
                token = token.lower()
            # if token.startswith("@user"):
            #     token = "@user"
            # if token.startswith("@http"):
            #     token = "http"
            tweet.append(token)
        test_data.append(tweet)
    all_possible_tags = read_file(in_tags_filename)
    uniqueTag = []
    #get a list of all the possible tags
    for tag in all_possible_tags:
        uniqueTag.append(tag.split("\n")[0])
    tag_result = []

    #viterbi algorithm
    for tweet in test_data:
        #send the sequence of tokens to the viterbi fn that will return us the
        #highest likelihood of tags for that particular sequence
        tag_seq = viterbiFn(tweet, uniqueTag, trans_prob_dic, bjw_dict)
        tag_result.append(tag_seq)

    fileVariable = open(out_predictions_filename, 'r+')
    fileVariable.truncate(0)
    fileVariable.close()
    with open(out_predictions_filename, 'w') as file:
        for lst in tag_result:
            for pred_tag in lst:
                file.write(f"{pred_tag}\n")
            file.write("\n")
    tag_result

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
    '''Viterbi predict2 accuracy:    1121/1378 = 0.8134978229317852'''
    '''
     We suggest to improve the viterbi algorithm by performing automatic preprocessing of the train data.
     We utilised computational methods i.e. Python regular expressions to identify certain patterns in the train data. 
     More specifically, we looked into the train data to identify specific types of tokens in the text data that
     may be difficult to tag accurately. 
     1. Websites. There were a substantial number of websites token in the train data that was associated with the ‘U’ tag. 
     We identified websites through the patterns like “https://“ or “www”. 
     By identifying these patterns using regular expression and replacing them with a generic token “hmm_website”,
     the accuracy of the POS tagger could be improved through reducing the number of unique tokens that the model has to learn.
     Moreover, the usage of a more generic tag for websites is able to help the model fit and generalise better to new websites it has not seen before, 
     increasing the likelihood of a correct tag prediction given a new website. 
     2. Similarly, usernames in Twitter are often preceded with the “@User” symbol, which is associated to the “@“ tag.
     We replaced these usernames with a generic “hmm_user”, and this could help the model to more easily identify 
     usernames and tag them to their correct tags even for new users it has not seen before, 
     thus increasing the probability of accurately predicting the tag, given a user token. 
     3. Twitter is known for its hashtags. Thus, we looked into the train data set to find many occurrences of ‘#’ 
     preceding a word that is tagged to the ‘#’ tag. We identified these hashtags and generalised them to “hmm_#”.
     This can help the model better recognise and tag words that are associated with #. 
     4. We also identified numeric values which are associated to the ‘$’ tag.
     For instance, dates, times, currency are replaced with the generic “hmm_num” to help the model better and more easily 
     recognise such tokens. 
     5. We also used contractions package in python to automatically help identify contractions and expanded them to long form
     For instance, we transformed words such as u to you, ppl to people, diff to different etc     
     Overall, through identifying and replacing the above types of tokens with a more generic tag,
     we aim to help the POS target to better handle unseen words and improve its accuracy on predicting tags of tokens 
     in the test data. Through reducing the number of unique and low frequency tokens in the data since all these tokens 
     (prior to preprocessing) have many variations, we lower the risk of overfitting and can improve the models’ 
     ability to generalise to new variations of the above 4 generalised tags and fixing the contractions to long form.
    '''
    trans_prob = pd.read_csv(in_trans_probs_filename, sep="\t")
    bjw = pd.read_csv(in_output_probs_filename, sep="\t")

    #unpack trans probability dict
    trans_prob_dic = {}
    for _, row in trans_prob.iterrows():
        if row['i'] not in trans_prob_dic:
            trans_prob_dic[row['i']] = {}
        trans_prob_dic[row['i']][row['j']] = row['prob']
    #unpack bjw probability dict
    bjw_dict = {}
    for _, row in bjw.iterrows():
        if row['token'] not in bjw_dict:
            bjw_dict[row['token']] = {}
        bjw_dict[row['token']][row['tag']] = row['bjw']

    corpus = read_file(in_test_filename)
    idx = [-1]
    for i in range(len(corpus)):
        # find the index of @
        element = corpus[i]
        if element == "\n":
            idx.append(i)
    corpus_user = []
    for i in range(len(idx)):
        num = idx[i]
        if (i == len(idx) - 1):
            tweet = corpus[num:]
        else:
            nextNum = idx[i + 1]
            tweet = corpus[num+1:nextNum]
        corpus_user.append(tweet)
    corpus_user
    test_data = []
    for lst in corpus_user:
        if len(lst) == 1:
            if lst[0] == "\n":
                continue
        tweet = []
        for ele in lst:
            token = ele.split("\t")[0].split("\n")[0]
            web_pattern = r'\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
            websites = re.findall(web_pattern, token)
            if websites:
                # this word is a website
                token = "hmm_website"
            else:
                token = token.lower()
                user_pattern = r'@user\w+'
                users = re.findall(user_pattern, token)
                if users:
                    # this word is a user
                    token = "hmm_user"
            if token.startswith("#"):
                token = "hmm_#"
            digits_pattern = r'\b\d+(?:\.\d+)?\b|\$\d+(?:,\d+)*(?:\.\d+)?|\d{1,2}:\d{2}(?::\d{2})?'
            digits = re.findall(digits_pattern, token)
            if digits:
                token = "hmm_num"

            contract_words = contractions.fix(token)
            if ((contract_words != token) and len(contract_words.split(" ")) == 1):
                # print("\n")
                # print(token)
                # print(contract_words)
                # print("\n")
                token = contract_words
            # token = spell(token)

            tweet.append(token)
        test_data.append(tweet)
    all_possible_tags = read_file(in_tags_filename)
    uniqueTag = []
    for tag in all_possible_tags:
        uniqueTag.append(tag.split("\n")[0])

    tag_result = []

    for tweet in test_data:
        tag_seq = viterbiFn(tweet, uniqueTag, trans_prob_dic, bjw_dict)
        tag_result.append(tag_seq)

    fileVariable = open(out_predictions_filename, 'r+')
    fileVariable.truncate(0)
    fileVariable.close()
    with open(out_predictions_filename, 'w') as file:
        for lst in tag_result:
            for pred_tag in lst:
                file.write(f"{pred_tag}\n")
            file.write("\n")
    tag_result




def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)

def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''

    ddir = '/Users/luyifan/Desktop/y3s1/bt3102/project/bt3102proj/projectfiles' #your working dir

    in_train_filename = f'{ddir}/twitter_train.txt'

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')
    
    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')

if __name__ == '__main__':
    run()
'''
By:
Lu Yi Fan A0233546E
Goh Kaitlyn Wen Jing A0239628R
Fong Weng Loke, Jesper A0233284H
Thia Jean Shuen, Summer A0239263B
'''
