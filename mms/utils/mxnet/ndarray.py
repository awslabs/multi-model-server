import mxnet as mx

def top_probability(data, labels, top=5):
    """Get top probability prediction from NDArray.

    Parameters
    ----------
    data : NDArray
        Data to be predicted
    labels : List
        List of class labels

    Returns
    -------
    List
        List of probability: class pairs in sorted order
    """
    sorted_prob = mx.nd.argsort(data[0], is_ascend=False)
    top_prob = map(lambda x: int(x.asscalar()), sorted_prob[0:top])
    return [{'probability': float(data[0, i].asscalar()), 'class': labels[i]}
            for i in top_prob]