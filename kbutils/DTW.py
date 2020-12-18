"""
DTW
input: seq1, seq2 (timeseries)
return: cost (distance or similarity)
reference : https://hamait.tistory.com/862pydoc
"""

import numpy as np
from numpy.linalg import norm
from functools import reduce

def dtw(seq1, seq2, n_window, distance, dist_small=True, edge_cost=None):
    """
    :param seq1: sentence vectorized series1
    :param seq2: sentence vectorized series2
    :param distance: distance calculation method
    """
    if dist_small:
        f_minmax = np.min
        if edge_cost is None:
            edge_cost = 1e6
        else:
            edge_cost = edge_cost
    else:
        f_minmax = np.max
        if edge_cost is None:
            edge_cost = -1e6
        else:
            edge_cost = edge_cost
        
    n_row = len(seq1)
    n_col = len(seq2)
    costs = edge_cost*np.ones((n_row,n_col),dtype=float)        

    # costs : row --> seq1, column --> seq2
    costs[0, 0] = distance(seq1[0], seq2[0])
    # calculate row and column side first
    for i in range(1, n_col):
        costs[0,i] = distance(seq1[0],seq2[i]) + costs[0,i-1]
    for i in range(1, n_row):
        costs[i,0] = distance(seq1[i],seq2[0]) + costs[i-1,0]
    # fill center, we can skip out of windows
    for i in range(1, n_row):
        for j in range(max(1, i-n_window), min(n_col, i+n_window)): 
            cost = distance(seq1[i],seq2[j])
            costs[i,j] = cost + f_minmax([costs[i-1,j],  \
                                        costs[i,j-1],  \
                                        costs[i-1,j-1]])

    # find the closest or farthest path
    paths = [(n_row-1,n_col-1,costs[n_row-1,n_col-1])]
    flag_finished = False
    i = n_row - 1
    j = n_col - 1
    # starts from (n_row-1, n_col), omit the starting point
    while flag_finished is False:
        prev_costs = [costs[i-1,j],costs[i-1,j-1],costs[i,j-1]]
        if i-j>n_window:
            prev_costs = [costs[i-1,j], costs[i-1,j-1], edge_cost]
        elif j-i>n_window:
            prev_costs = [edge_cost , costs[i-1,j-1], costs[i,j-1]]
        move_cost = f_minmax(prev_costs)
        pre_pos = prev_costs.index(move_cost)
        
        if pre_pos==0:
            i = i-1
            paths.append((i,j,move_cost))            
        elif pre_pos==1:
            i = i-1
            j = j-1
            paths.append((i,j,move_cost))
        else:
            j = j-1
            paths.append((i,j,move_cost))

        if i<=0 and j<=0:
            flag_finished = True
        elif i<=0:
            for j1 in range(j-1,-1, -1):
                paths.append((0, j1, costs[0, j1]))
            flag_finished = True
        elif j<=0:
            for i1 in range(i-1, -1, -1):
                paths.append((i1, 0, costs[i1, 0]))
            flag_finished = True
    paths.reverse()
    paths_seq1 = [x[0] for x in paths]
    paths_seq2 = [x[1] for x in paths]
    
    return paths[-1][2]/len(paths_seq1), paths_seq1, paths_seq2

def cos_sim1_non_zero(vec1, vec2):
    # it seems like vec1 and vec2 are 2d array
    # gives offset and get inverse value for distance
    ret = np.inner(vec1, vec2)/(norm(vec1)*norm(vec2))
    return 1/(ret+1.5)

if __name__ == '__main__':
    def distance(p1,p2):
        return (p1-p2)**2
    
    dtw(test=True, n_window=2, distance=distance)
