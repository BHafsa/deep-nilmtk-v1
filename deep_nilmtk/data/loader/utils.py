def target_generator( type, in_size, out_size, point_position=None):
    """
    initialise how the target indices are generated
    :param type: type of the target model
    :param point_position: position of the point for seq2point models
    :return:  function to calculate the target indices
    """
    if type == 'seq2seq' or type == 'seq2quantile':
        assert in_size == out_size, "Target sequence length should be equal to the input sequence"
        return lambda x: x
    elif type == 'seq2point':
        assert point_position is not None, "The point position is not specified"
        assert out_size == 1, "the target sequence length should be 1"
        if point_position == 'last_position':
            return lambda x: x[-1]
        elif point_position == 'mid_position':
            return lambda x: x[len(x) // 2]
        else:
            raise Exception(f"The specified position is not recognised, Expected [last_position, mid_position] found {point_position}")
    elif type == 'seq2subseq':
        assert out_size < in_size, "Target sequence length should less than the input sequence"
        diff = in_size - out_size
        return lambda x: x[diff // 2: -diff // 2]
    else:
        raise Exception(
            "The specified training approach is not recognised, please try [seq2point, seq2point, seq2subseq]")