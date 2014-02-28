import dual_affinity

def CV_predict(new_data,reference_data,reference_field,row_tree,alpha,beta):
    """
    OK, we have organized and predicted reference_field from reference_data
    (those should be the same size)
    Now we want to fit the columns of new_data (as observations) into the 
    organization.
    So let's just calculate the EMD from the new observations to the old
    and average them with a kernel smoother.
    """
    emds = dual_affinity.calc_emd_ref(reference_data, new_data, row_tree,
                                      alpha, beta)
    