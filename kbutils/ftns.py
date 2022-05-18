import numpy as np

def batch_shuffle(i_ix, f_ix, batch_size, n_times=1):
    rets = []
    for ii in range(n_times):
        ret = batch_shuffle_single(i_ix, f_ix, batch_size)
        rets.append(ret)
    return rets

def batch_shuffle_single(i_ix, f_ix, batch_size):
    ids1 = range(i_ix, f_ix)
    ids2 = []
    for ii, xx in enumerate(ids1):
        if ii%batch_size == 0:
            ids2.append([])
        ids2[-1].append(xx)

    ixs = list(range(len(ids2)))
    np.random.shuffle(ixs)
    ids3 = [ids2[i] for i in ixs]

    ret = []
    for ii in range(len(ids3)):
        for yy in ids3[ii]:
            ret.append(yy)
    return ret

def moving_acc(outs_ratio0, tgts_ratio0, outs_ratio, tgts_ratio,
               mean_vals, range_vals, past_prices, tgts, flag_l1 = False):
    """
    parameters are numpy array
    """
    outs_ratio2 = np.append(outs_ratio0, outs_ratio)
    tgts_ratio2 = np.append(tgts_ratio0, tgts_ratio)

    n_tgts_ratio = len(outs_ratio0)
    xsp = cp.copy(outs_ratio)
    xsp2 = cp.copy(outs_ratio)
    reg = LinearRegression(positive=True)
    for ii in range(len(tgts_ratio)):
        i_xs = ii
        f_xs = n_tgts_ratio + ii
        xs, ys = outs_ratio2[i_xs:f_xs], tgts_ratio2[i_xs:f_xs]
        #xs, ys = outs_ratio2[:f_xs], tgts_ratio2[:f_xs]
        xs = xs.reshape(-1, 1)
        weights = 1+np.abs(ys*10)
        ys = ys.reshape(-1, 1)
        regre = reg.fit(xs, ys, sample_weight=weights)
        xin = outs_ratio[ii].reshape(-1,1)
        xp = regre.predict(xin)[0]
        score = reg.score(xs, ys, sample_weight=weights)
        xsp2[ii] = xp
        
        if score < 0.2:
            xp = outs_ratio[ii]
        else:
            xp = (1-score)*outs_ratio[ii] + score*xp
        xsp[ii] = xp

    bacc = balanced_accuracy_score(tgts_ratio>0, xsp>0)
    prices = xsp/10 * past_prices + past_prices
    outsp = (prices - mean_vals)/range_vals
    loss = np.mean(np.abs(outsp-tgts))
    #bacc = balanced_accuracy_score(tgts_ratio>0, xss3>0)
    return bacc, loss, prices

def dim_num(x):
    x2 = x*3//2
    if x2%2==0:
        return x2
    else:
        return x2+1    
    
def sort_by_ratio(dates, mkt_data):
    ixs = sorted(range(len(dates)), key = lambda ix: np.abs(mkt_data[dates[ix]][5]), reverse=True)
    return [dates[ix1] for ix1 in ixs], ixs

def intervals(size_data, ix_test0
              , pred_period, size_valid, size_test):
    """
    fx_train : end point of train
    ix_valid : starting point of validation
    ix_test : starting point of validation
    """
    ix_test = ix_test0
    flag_end = False
    while True:
        ix_valid = ix_test - size_valid
        fx_train = ix_valid - pred_period
        fx_test = ix_test + size_test
        if fx_test >= size_data:
            yield fx_train, ix_valid, ix_test, size_data
            break
        else:
            yield fx_train, ix_valid, ix_test, fx_test
        ix_test += size_test