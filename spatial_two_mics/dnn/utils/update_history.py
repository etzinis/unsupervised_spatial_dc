"""!
@brief History and callback update functions

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

def values_update(list_of_pairs,
                  history_dic,
                  update_mode='batch'):
    """! Update the history dictionary for each key, value pair
    INPLACE and stores values for batch and epoch
    :param update_mode: In batch mode the values of the specific key
    would be summed and in epoch mode would be averaged throughout
    the batches.
    :param list_of_pairs: list of tuples e.g. [('loss', 0.9987), ...,]
    :param history_dic: a dictionary that we want to keep track for
    a metric under all epochs
    :return: history_dic updated with all the appropriate values for
    batch and epoch
    """
    if update_mode == 'batch':
        for k, v in list_of_pairs:
            if not k+"_batch_total" in history_dic:
                history_dic[k] = []
                history_dic[k+"_batch_total"] = v
                history_dic[k + '_batch_counter'] = 1
            else:
                history_dic[k + "_batch_total"] += v
            history_dic[k+'_batch_counter'] += 1
    elif update_mode == 'epoch':
        for k, v in list_of_pairs:
            history_dic[k].append(history_dic[k + "_batch_total"] /
                                  history_dic[k + '_batch_counter'])
            history_dic[k + "_batch_total"] = 0.
            history_dic[k + '_batch_counter'] = 0
    else:
        raise NotImplementedError('Please use an update mode of epoch '
                                  'or batch')

    return history_dic
