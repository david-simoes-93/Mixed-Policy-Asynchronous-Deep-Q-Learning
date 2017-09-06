def projection(PI, threshold=0):
    #print(PI)
    # 1- the unit vector is perpendicular to the 1 surface. Using this along with the
    # 	passed point P0 we get a parametric line equation:
    #   P = P0 + t * I, where t is the parameter and I is the unit vector.
    # 	the unit of projection has \sum P = 1 = \sum P0 + nt, where n is the dimension of P0
    # 	hence the point of projection, P' = P0 + ( (1 - \sum P0) / n ) I
    # * compute sum
    t = sum(PI)
    #print(t)
    # * compute t
    t = (1.0 - t) / len(PI)

    # * compute P'
    for i in range(len(PI)):
        PI[i] += t

    # 2- if forall p in P', p >=0 (and consequently <=1), we found the point.
    #	other wise, pick a negative dimension d, make it equal zero while decrementing
    #	other non zero dimensions. repeat until no negatives remain.
    done = False
    while not done:
        # comulate negative dimensions
        # and count positive ones. note that there must be at least
        # one positive dimension
        n = 0
        excess = 0
        for i in range(len(PI)):
            if PI[i] < threshold:
                excess += threshold-PI[i]
                PI[i] = threshold
            elif PI[i] > threshold:
                n += 1

        # none negative? then done
        if excess == 0:
            done = True
        else:
            # otherwise decrement by equal steps
            for i in range(len(PI)):
                if PI[i] > threshold:
                    PI[i] -= excess / n
    #print(PI)
