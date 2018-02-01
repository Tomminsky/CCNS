import six
import numpy as np

# this script is used in combination with the actual word embedding script
# this was to ensure only minor changes to the original script
# functions below will be imported by the actual word embedding script

def get_dictionary(namefile):
    with open(namefile) as f:
        dict_={line.replace('"', '').replace('\n', ''): int(v) for v, line in enumerate(f)}
    dict_.update({'<eol>': len(dict_)+1})
    return dict_


def get_title(namefile,dictionary):

    with open(namefile) as file:
        lines = ''
        next(file)
        for line in file:
            lines = lines + line.replace(',0', '').replace('\n', '').replace('"','') + ',' + str(dictionary.get('<eol>')) + ','

    titles=np.array(map(int,lines[:-1].split(',')))

    eolidx = [i for i, x in enumerate([titles[y] == dictionary.get('<eol>') for y in range(len(titles))]) if x]

    size_titles = len(eolidx)-1
    idx_shuff_titles = np.random.permutation(size_titles)

    sel_for_val = idx_shuff_titles[range(int(round(size_titles / 10.)))]

    sel_for_title = idx_shuff_titles[range(int(round(size_titles / 10.)), size_titles)]

    eol_end = np.array([x + 1 for x in eolidx[1:]])[sel_for_val]
    eol_start = np.array([x + 1 for x in eolidx[:-1]])[sel_for_val]

    val = []
    for n in range(len(eol_end)):
        val = val + list(titles[range(eol_start[n], eol_end[n])])

    eol_end = np.array([x + 1 for x in eolidx[1:]])[sel_for_title]
    eol_start = np.array([x + 1 for x in eolidx[:-1]])[sel_for_title]

    train = []
    for n in range(len(eol_end)):
        train = train + list(titles[range(eol_start[n], eol_end[n])])

    return train,val


def disp_title(all_title_idx,dictionary,number_of_title):
    """
        Displays the nth title in all_title_idx represented as indices using the dictionary
        Titles are seperated by coded <eol> (end of line)
    """
    idx_end_of_line = dictionary.get('<eol>')
    index2word = {wid: word for word, wid in six.iteritems(dictionary)}
    eolidx = [i for i, x in enumerate([all_title_idx[y] == idx_end_of_line for y in range(len(all_title_idx))]) if x]

    return [index2word[all_title_idx[x]] for x in range(eolidx[number_of_title]+1,eolidx[number_of_title+1]+1)]