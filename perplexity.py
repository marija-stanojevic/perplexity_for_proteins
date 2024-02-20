import pandas as pd
import numpy as np


# supports sequences of length up to 9999
def make_string_len_four(k: int):
    one = k // 1000
    two = k % 1000 // 100
    three = k % 100 // 10
    four = k % 10
    return str(one) + str(two) + str(three) + str(four)


def perplexity_from_prob(prob_seq, generated_seq:pd.Series):
    pp_per_sample = []
    N = len(list(generated_seq[0]))
    for prob in prob_seq:
        pp_per_sample.append(pow(prob, -1 / N))
    return [np.mean(pp_per_sample), pp_per_sample]


def sum_sequences(prob_seq1, prob_seq2):
    prob_seq = []
    for k in range(len(prob_seq1)):
        prob_seq.append(prob_seq1[k] + prob_seq2[k])
    return prob_seq


def unigram_probs(corpus:pd.Series, generated_seq:pd.Series):
    dict = {}
    for _, sample in corpus.iteritems():
        amino_acids = list(sample)
        for k in range(len(amino_acids)):
            id = amino_acids[k] + make_string_len_four(k)  # amino acid and its position
            if id in dict.keys():
                dict[id] += 1
            else:
                dict[id] = 1
    for k in dict:
        dict[k] /= len(corpus)
    prob_seq = []
    for sample in generated_seq:
        amino_acids = list(sample)
        prob = 1
        for k in range(len(amino_acids)):
            id = amino_acids[k] + make_string_len_four(k)
            if id in dict.keys():
                prob *= dict[id]
        prob_seq.append(prob)
    return prob_seq


def bigram_probs(corpus:pd.Series, generated_seq:pd.Series, type:'both'):
    """
       Type can have vlaue: 'left', 'right', 'both'. Left means that senquence dependEncy goes from left to right p(x2|x1).
       Right means that sequence dependency goes from right to left p(x1|x2).
       Both means that dependency goes from both sides p(x1,x2) at given position.
       """
    dict = {}
    cond_dict = {}
    for _, sample in corpus.iteritems():
        amino_acids = list(sample)
        for k in range(len(amino_acids) - 1):
            pos = make_string_len_four(k)
            if type == 'left':
                id = amino_acids[k] + pos
                if id in cond_dict:
                    cond_dict[id] += 1
                else:
                    cond_dict[id] = 1
            elif type == 'right':
                id = amino_acids[k + 1] + pos
                if id in cond_dict:
                    cond_dict[id] += 1
                else:
                    cond_dict[id] = 1
            else:
                id = amino_acids[k] + amino_acids[k + 1] + pos
                if id in cond_dict:
                    cond_dict[id] += 1
                else:
                    cond_dict[id] = 1
            id = amino_acids[k] + amino_acids[k + 1] + pos  # amino acid and its position
            if id in dict.keys():
                dict[id] += 1
            else:
                dict[id] = 1
    for k in dict:
        if type == 'left':
            cond_id = k[0] + k[2:6]
        elif type == 'right':
            cond_id = k[1] + k[2:6]
        else:
            cond_id = k
        dict[k] /= cond_dict[cond_id]
    prob_seq = []
    for sample in generated_seq:
        amino_acids = list(sample)
        prob = 1
        for k in range(len(amino_acids) - 1):
            id = amino_acids[k] + amino_acids[k + 1] + make_string_len_four(k)
            if id in dict.keys():
                prob *= dict[id]
        prob_seq.append(prob)
    return prob_seq


def trigram_probs(corpus:pd.Series, generated_seq:pd.Series, type:'both'):
    """
       Type can have vlaue: 'left', 'right', 'both'. Left means that senquence dependEncy goes from left to right p(x2|x1).
       Right means that sequence dependency goes from right to left p(x1|x2).
       Both means that dependency goes from both sides p(x1,x2) at given position.
       """
    dict = {}
    cond_dict = {}
    for _, sample in corpus.iteritems():
        amino_acids = list(sample)
        for k in range(len(amino_acids) - 2):
            pos = make_string_len_four(k)
            if type == 'left':
                id = amino_acids[k] + amino_acids[k + 1] + pos
                if id in cond_dict:
                    cond_dict[id] += 1
                else:
                    cond_dict[id] = 1
            elif type == 'right':
                id = amino_acids[k + 1] + amino_acids[k + 2] + pos
                if id in cond_dict:
                    cond_dict[id] += 1
                else:
                    cond_dict[id] = 1
            else:
                id = amino_acids[k] + amino_acids[k + 1] + amino_acids[k + 2] + pos
                if id in cond_dict:
                    cond_dict[id] += 1
                else:
                    cond_dict[id] = 1
            id = amino_acids[k] + amino_acids[k + 1] + amino_acids[k + 2] + pos  # amino acid and its position
            if id in dict.keys():
                dict[id] += 1
            else:
                dict[id] = 1
    for k in dict:
        if type == 'left':
            cond_id = k[0:2] + k[3:7]
        elif type == 'right':
            cond_id = k[1:3] + k[3:7]
        else:
            cond_id = k
        dict[k] /= cond_dict[cond_id]
    prob_seq = []
    for sample in generated_seq:
        amino_acids = list(sample)
        prob = 1
        for k in range(len(amino_acids) - 2):
            id = amino_acids[k] + amino_acids[k + 1] + amino_acids[k + 2] + make_string_len_four(k)
            if id in dict.keys():
                prob *= dict[id]
        prob_seq.append(prob)
    return prob_seq


def positional_perplexity(corpus: pd.Series, generated_seq:pd.Series, type:'both'):
    """
        Type can have vlaue: 'left', 'right', 'both'. Left means that senquence dependEncy goes from left to right
        Right means that sequence dependency goes from right to left. Both means that dependency goes from both sides
        """
    dict = {}
    cond_dict = {}
    for _, sample in corpus.iteritems():
        amino_acids = list(sample)
        id = ''
        if type == 'left':
            for k in range(len(amino_acids)):
                if id in cond_dict:
                    cond_dict[id] += 1
                else:
                    cond_dict[id] = 1
                id += amino_acids[k] + make_string_len_four(k)  # amino acid and its position
                if id in dict.keys():
                    dict[id] += 1
                else:
                    dict[id] = 1
        elif type == 'right':
            for k in range(len(amino_acids) - 1, - 1, -1):
                if id in cond_dict:
                    cond_dict[id] += 1
                else:
                    cond_dict[id] = 1
                id += amino_acids[k] + make_string_len_four(k)  # amino acid and its position
                if id in dict.keys():
                    dict[id] += 1
                else:
                    dict[id] = 1
        else:
            for k in range(len(amino_acids), - 1, -1):
                id = sample[0:k] + sample[k+1:]
                if id in cond_dict:
                    cond_dict[id] += 1
                else:
                    cond_dict[id] = 1
                id = sample + make_string_len_four(k)  # amino acid and its position
                if id in dict.keys():
                    dict[id] += 1
                else:
                    dict[id] = 1
    for k in dict:
        if type == 'left':
            cond_id = k[:-5]
        elif type == 'right':
            cond_id = k[:-5]
        else:
            pos = int(k[-4:])
            cond_id = k[:pos] + k[pos+1:-4]
        dict[k] /= cond_dict[cond_id]
    pp_per_sample = []
    for sample in generated_seq:
        amino_acids = list(sample)
        prob = 1
        id = ''
        for k in range(len(amino_acids)):
            if type == 'left':
                id += amino_acids[k] + make_string_len_four(k)
            elif type == 'right':
                id += amino_acids[k] + make_string_len_four(k)
            else:
                id += sample + make_string_len_four(k)
            if id in dict.keys():
                prob *= dict[id]
        N = len(amino_acids)
        pp_per_sample.append(pow(prob, -1 / N))
    return [np.mean(pp_per_sample), pp_per_sample]


def perplexity(corpus_filename:str, generated_seq_filename:str, type: str = 'unigram', order: str = 'left'):
    """Dataframe rows are currated protein sequences.
       One sequence is a string of letters representing different amino-acids.
       Type allows you to choose the formula. Options are:
            1) 'unigram'; 2) 'bigram'; 3) 'positional'; 4) 'energy_eq': it's a sum of 1 and 2
       Order shows how to condition data: 1) 'left' for left to right;
       2) 'right' for right to left; 3) 'both' for non-directional
    """
    corpus = pd.read_csv(corpus_filename, index_col=False, header=None)
    generated_seq = pd.read_csv(generated_seq_filename, index_col=False, header=None)

    corpus = corpus[0]
    generated_seq = generated_seq[0]
    pp_score = 0
    pp_per_sample = []
    if type == 'unigram':
        prob_seq = unigram_probs(corpus, generated_seq)
        pp_score, pp_per_sample = perplexity_from_prob(prob_seq, generated_seq)
    if type == 'bigram':
        prob_seq = bigram_probs(corpus, generated_seq, order)
        pp_score, pp_per_sample = perplexity_from_prob(prob_seq, generated_seq)
    if type == 'trigram':
        prob_seq = trigram_probs(corpus, generated_seq, order)
        pp_score, pp_per_sample = perplexity_from_prob(prob_seq, generated_seq)
    if type == 'positional':
        pp_score, pp_per_sample = positional_perplexity(corpus, generated_seq, order)
    if type == 'energy_eq':
        prob_seq1 = unigram_probs(corpus, generated_seq)
        prob_seq2 = bigram_probs(corpus, generated_seq, order)
        prob_seq = sum_sequences(prob_seq1, prob_seq2)
        pp_score, pp_per_sample = perplexity_from_prob(prob_seq, generated_seq)

    print(pp_score)
    return pp_score, pp_per_sample


file1 = 'corpus.csv'
file2 = 'gen_text.csv'
perplexity(file1, file2, 'trigram', 'both')
