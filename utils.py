import numpy as np


def precision_at_k(y_true, y_pred, k=12):
    """ Computes Precision at k for one sample

    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations

    Returns
    _______
    score: double
           Precision at k
    """
    intersection = np.intersect1d(y_true, y_pred[:k])
    return len(intersection) / k



def rel_at_k(y_true, y_pred, k=12):
    """ Computes Relevance at k for one sample

    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations

    Returns
    _______
    score: double
           Relevance at k
    """
    if y_pred[k-1] in y_true:
        return 1
    else:
        return 0

def average_precision_at_k(y_true, y_pred, k=12):
    """ Computes Average Precision at k for one sample

    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations

    Returns
    _______
    score: double
           Average Precision at k
    """
    ap = 0.0
    rel_counter = 0
    for i in range(1, k+1):
        ap += precision_at_k(y_true, y_pred, i) * rel_at_k(y_true, y_pred, i)
        rel_counter += rel_at_k(y_true, y_pred, i)
    # return ap / min(k, len(y_true))
    if rel_counter == 0:
        return 0
    return ap / rel_counter

def mean_average_precision(y_true, y_pred, k=12):
    """ Computes MAP at k

    Parameters
    __________
    y_true: np.array
            2D Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            2D Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations

    Returns
    _______
    score: double
           MAP at k
    """

    return np.mean([average_precision_at_k(gt, pred, k)
                    for gt, pred in zip(y_true, y_pred)])


def mAP_at_k(D, imgLab, gt, rank=1, posonly="False"):

    _, idx = D.topk(rank[-1],  dim=1 )
    preds = np.array([imgLab[i].numpy() for i in idx])

    if posonly == "True":
        return mean_average_precision(gt[:10],preds[:10], k= rank[-1]),idx
    elif posonly == "divide_by_class" : 
        return [mean_average_precision(gt[:10],preds[:10], k= rank[-1]),mean_average_precision(gt[10:20],preds[10:20], k= rank[-1]),mean_average_precision(gt[20:30],preds[20:30], k= rank[-1]),mean_average_precision(gt[30:40],preds[30:40], k= rank[-1]),mean_average_precision(gt[40:50],preds[40:50], k= rank[-1]),mean_average_precision(gt,preds, k= rank[-1])],idx
    else :
        return mean_average_precision(gt, preds, k= rank[-1]), idx










if __name__ == '__main__':

    gt = np.array(['a', 'b', 'c', 'd', 'e'])

    preds1 = np.array(['b', 'c', 'a', 'd', 'e'])
    preds2 = np.array(['a', 'b', 'c', 'd', 'e'])
    preds3 = np.array(['f', 'b', 'c', 'd', 'e'])
    preds4 = np.array(['a', 'f', 'e', 'g', 'b'])
    preds5 = np.array(['a', 'f', 'c', 'g', 'b'])
    preds6 = np.array(['d', 'c', 'b', 'a', 'e'])


    assert precision_at_k(gt, preds1, k=1) == 1.0
    assert precision_at_k(gt, preds2, k=1) == 1.0
    assert precision_at_k(gt, preds3, k=1) == 0.0
    assert precision_at_k(gt, preds4, k=2) == 1/2
    assert precision_at_k(gt, preds5, k=3) == 2/3
    assert precision_at_k(gt, preds6, k=3) == 3/3


    assert rel_at_k(gt, preds1, k=1) == 1.0
    assert rel_at_k(gt, preds2, k=1) == 1.0
    assert rel_at_k(gt, preds3, k=1) == 0.0
    assert rel_at_k(gt, preds4, k=2) == 0.0
    assert rel_at_k(gt, preds5, k=3) == 1.0
    assert rel_at_k(gt, preds6, k=3) == 1.0

    # Bu doğru mu, average_precision_at_k fonksiyonunda eksiklik saptamıştık!
    assert average_precision_at_k(gt, preds1, k=1) == 1.0
    assert average_precision_at_k(gt, preds2, k=1) == 1.0
    assert average_precision_at_k(gt, preds3, k=1) == 0.0
    assert average_precision_at_k(gt, preds4, k=2) == 0.5
    assert average_precision_at_k(gt, preds5, k=3) == 0.5555555555555555
    assert average_precision_at_k(gt, preds6, k=3) == 1.0

    print(average_precision_at_k(gt, preds1, k=4))
    print(average_precision_at_k(gt, preds2, k=4))
    print(average_precision_at_k(gt, preds3, k=4))
    print(average_precision_at_k(gt, preds4, k=4))
    print(average_precision_at_k(gt, preds5, k=4))
    print(average_precision_at_k(gt, preds6, k=4))
    
    y_true = np.array([gt, gt, gt, gt, gt, gt])
    y_pred = np.array([preds1, preds2, preds3, preds4, preds5, preds6])

    mean_average_precision(y_true, y_pred, k=4)