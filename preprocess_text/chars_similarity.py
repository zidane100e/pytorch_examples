import re

def get_chars(title):
    korean = "[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]"
    chars = []
    for char in title:
        if re.search(korean, char):
            chars.append(char)
    return ''.join(chars)

def title2dic(title):
    chars_d = {}
    for char in title:
        chars_d.setdefault(char, 0)
        chars_d[char] += 1
    return chars_d

def count_char_dic(dic):
    count = 0
    for val in dic.values():
        count += 1
    return count

def subtract_char_dic(dic1, dic2):
    dic3 = {}
    keys2 = dic2.keys()
    for key, val in dic1.items():
        if key in keys2:
            if val - dic2[key] != 0:
                dic3[key] = val - dic2[key]
        else:
            dic3[key] = val
    return dic3

def distance_char_dic(dic1, dic2):
    dic3 = subtract_char_dic(dic1, dic2)
    n_dic1 = count_char_dic(dic1)
    n_dic3 = count_char_dic(dic3)
    return 1.0*n_dic3/n_dic1

def title_char_distance(titles, id1, ids, limit=0.3):
    title1 = titles[id1]
    title1 = get_chars(title1)
    dic1 = title2dic(title1)
    
    simils_ids = []
    for id2 in ids:
        title2 = titles[id2]
        title2 = get_chars(title2)
        dic2 = title2dic(title2)
        #print('--', dic1, dic2, title1, '##', title2, id2)
        dist = distance_char_dic(dic1, dic2)
        if dist < limit:
            simils_ids.append(id2)
    return simils_ids
    
def similar_titles(titles, ids, groups):
    """
    for a title compare char dist for all other titles
    find similar items and subtract them
    operate recursively
    :titles: input text
    :ids: only see ids of text
    :param groups: this modifies groups. gives all groups, You need to select one from each groups
    """
    if len(ids)<1: 
        return groups
    id1 = ids[0]
    ids = ids[1:]
    
    similars = title_char_distance(titles, id1, ids)
    groups.append([id1] + similars)
    ids = list(set(ids) - set(similars))
    similar_titles(titles, ids, groups)
    
    
if __name__ == '__main__':
    # titles = df['title'].tolist()
    title1 = "뱅크론ㆍ하이일드펀드, ‘달러강세’ 엔진에 好好"
    title2 = "뱅크론·하이일드펀드, '달러강세' 엔진에 好好"
    title3 = "빨간불 켜진 한국 경제…올해 3% 성장 의구심 고조"
    titles = [title1, title2, title3]
    ids = list(range(3))
    groups = []
    similar_titles(titles, ids, groups)