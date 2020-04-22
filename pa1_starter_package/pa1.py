# Starter code for CS 165B HW2 Spring 2019

def train(training_input):
    # Get the dimentionality, the sample number of class A, class B, and class C
    dim, na, nb, nc = training_input[0]
    
    # Compute the centroid for class A
    ca = [0, 0, 0]
    for i in range(na):
        ca[0] += training_input[i + 1][0]
        ca[1] += training_input[i + 1][1]
        ca[2] += training_input[i + 1][2]
    ca[0] = ca[0] / na
    ca[1] = ca[1] / na
    ca[2] = ca[2] / na

    # Compute the centroid for class B
    cb = [0, 0, 0]
    for i in range(nb):
        cb[0] += training_input[i + na + 1][0]
        cb[1] += training_input[i + na + 1][1]
        cb[2] += training_input[i + na + 1][2]
    cb[0] = cb[0] / nb
    cb[1] = cb[1] / nb
    cb[2] = cb[2] / nb

    # Compute the centroid for class C
    cc = [0, 0, 0]
    for i in range(nc):
        cc[0] += training_input[i + nb + na + 1][0]
        cc[1] += training_input[i + nb + na + 1][1]
        cc[2] += training_input[i + nb + na + 1][2]
    cc[0] = cc[0] / nc
    cc[1] = cc[1] / nc
    cc[2] = cc[2] / nc

    # Compute classifier A/B
    w_ab = [ca[0] - cb[0], ca[1] - cb[1], ca[2] - cb[2]]
    half_t_ab = [(ca[0] + cb[0])*0.5, (ca[1] + cb[1])*0.5, (ca[2] + cb[2])*0.5]
    t_ab = half_t_ab[0] * w_ab[0] + half_t_ab[1] * w_ab[1] + half_t_ab[2] * w_ab[2]

    # Compute classifier B/C
    w_bc = [cb[0] - cc[0], cb[1] - cc[1], cb[2] - cc[2]]
    half_t_bc = [(cc[0] + cb[0])*0.5, (cc[1] + cb[1])*0.5, (cc[2] + cb[2])*0.5]
    t_bc = half_t_bc[0] * w_bc[0] + half_t_bc[1] * w_bc[1] + half_t_bc[2] * w_bc[2]

    # Compute classifier A/C
    w_ac = [ca[0] - cc[0], ca[1] - cc[1], ca[2] - cc[2]]
    half_t_ac = [(cc[0] + ca[0])*0.5, (cc[1] + ca[1])*0.5, (cc[2] + ca[2])*0.5]
    t_ac = half_t_ac[0] * w_ac[0] + half_t_ac[1] * w_ac[1] + half_t_ac[2] * w_ac[2]

    result = [w_ab, t_ab, w_bc, t_bc, w_ac, t_ac]
    return result

def test(testing_input, classifiers):
    dim, na, nb, nc = testing_input[0]
    w_ab = classifiers[0]
    t_ab = classifiers[1]
    w_bc = classifiers[2]
    t_bc = classifiers[3]
    w_ac = classifiers[4]
    t_ac = classifiers[5]

    # Test class A
    tp = [0, 0, 0]
    tn = [0, 0, 0]
    fp = [0, 0, 0]
    fn = [0, 0, 0]
    for i in range(na):
        xt_ab = testing_input[i + 1][0] * w_ab[0] + testing_input[i + 1][1] * w_ab[1] + testing_input[i + 1][2] * w_ab[2]
        xt_ac = testing_input[i + 1][0] * w_ac[0] + testing_input[i + 1][1] * w_ac[1] + testing_input[i + 1][2] * w_ac[2]
        if xt_ab >= t_ab and xt_ac >= t_ac: # Actual A, predicted A
            tp[0] += 1
            tn[1] += 1
            tn[2] += 1
        elif xt_ab < t_ab: # Actual A, predicted B
            fn[0] += 1
            fp[1] += 1
            tn[2] += 1
        else: # Actual A, predicted C
            fn[0] += 1
            tn[1] += 1
            fp[2] += 1
    
    for i in range(nb):
        xt_ab = testing_input[i + na + 1][0] * w_ab[0] + testing_input[i + na + 1][1] * w_ab[1] + testing_input[i + na + 1][2] * w_ab[2]
        xt_bc = testing_input[i + na + 1][0] * w_bc[0] + testing_input[i + na + 1][1] * w_bc[1] + testing_input[i + na + 1][2] * w_bc[2]
        if xt_bc >= t_bc and xt_ab < t_ab: # Actual B, predicted B
            tn[0] += 1
            tp[1] += 1
            tn[2] += 1
        elif xt_ab >= t_ab: # Actual B, predicted A
            fp[0] += 1
            fn[1] += 1
            tn[2] += 1
        elif xt_bc < t_bc: # Actual B, predicted C
            tn[0] += 1
            fn[1] += 1
            fp[2] += 1

    for i in range(nc):
        xt_ac = testing_input[i + na + nb + 1][0] * w_ac[0] + testing_input[i + na + nb + 1][1] * w_ac[1] + testing_input[i + na + nb + 1][2] * w_ac[2]
        xt_bc = testing_input[i + na + nb + 1][0] * w_bc[0] + testing_input[i + na + nb + 1][1] * w_bc[1] + testing_input[i + na + nb + 1][2] * w_bc[2]
        if xt_ac < t_ac and xt_bc < t_bc: # Actual C, prediceted C
            tn[0] += 1
            tn[1] += 1
            tp[2] += 1
        elif xt_ac >= t_ac: # Actual C, predicted A
            fp[0] += 1
            tn[1] += 1
            fn[2] += 1
        elif xt_bc >= t_bc: # Actual C, predicted B
            tn[0] += 1
            fp[1] += 1
            fn[2] += 1

    # Test results for class A
    tpr_a = tp[0]/na
    fpr_a = fp[0]/(nb+nc)
    error_rate_a = (fp[0] + fn[0])/(na+nb+nc)
    accuracy_a = (tp[0] + tn[0])/(na+nb+nc)
    precision_a = tp[0]/(tp[0]+fp[0])

    # Test results for class B
    tpr_b = tp[1]/nb
    fpr_b = fp[1]/(na+nc)
    error_rate_b = (fp[1] + fn[1])/(na+nb+nc)
    accuracy_b = (tp[1] + tn[1])/(na+nb+nc)
    precision_b = tp[1]/(tp[1]+fp[1])

    # Test results for class C
    tpr_c = tp[2]/nc
    fpr_c = fp[2]/(nb+na)
    error_rate_c = (fp[2] + fn[2])/(na+nb+nc)
    accuracy_c = (tp[2] + tn[2])/(na+nb+nc)
    precision_c = tp[2]/(tp[2]+fp[2])

    tpr = (tpr_a + tpr_b + tpr_c) / 3
    fpr = (fpr_a + fpr_b + fpr_c) / 3
    error_rate = (error_rate_a + error_rate_b + error_rate_c)/3
    accuracy = (accuracy_a + accuracy_b + accuracy_c) / 3
    precision = (precision_a + precision_b + precision_c) / 3
    result = {"tpr": tpr, "fpr":fpr, "error_rate":error_rate, "accuracy":accuracy, "precision":precision}
    return result

def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.
    You are permitted to use the numpy library but you must write
    your own code for the linear classifier.

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED

        Example:
            return {
                "tpr": #your_true_positive_rate,
                "fpr": #your_false_positive_rate,
                "error_rate": #your_error_rate,
                "accuracy": #your_accuracy,
                "precision": #your_precision
            }
    """
    result = {}
    classifiers = train(training_input)
    result = test(testing_input, classifiers)

    return result


    # TODO: IMPLEMENT
    pass
