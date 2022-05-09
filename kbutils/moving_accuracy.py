from sklearn.linear_model import LinearRegression

def moving_acc(outs_ratio0, tgts_ratio0, outs_ratio, tgts_ratio,

               mean_vals, range_vals, past_prices, tgts):

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

        #if True:

            xp = outs_ratio[ii]

        xsp[ii] = xp

    bacc = balanced_accuracy_score(tgts_ratio>0, xsp>0)

    prices = xsp/10 * past_prices + past_prices

    outsp = (prices - mean_vals)/range_vals

    loss = np.mean((outsp-tgts)**2)

    #bacc = balanced_accuracy_score(tgts_ratio>0, xss3>0)

    return bacc, loss, prices

 

#=================================

 

mean_vals = retsv[11]

range_vals = retsv[12]

past_prices = retsv[13]

tgtsv = retsv[14]

baccp, lossp, outsp = moving_acc(retsv0[5], retsv0[6], 

                                  retsv[5], retsv[6],

                                  mean_vals, range_vals, past_prices, tgtsv

                                 )
