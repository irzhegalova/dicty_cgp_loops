import numpy as np

# %% fuctions
def compute_flames(stack, flank=10000, resolution=2000, remove_na=True,
                   pad=3,
                   left=True):
    """
    create 3D array with the flame (right if left = False or left if left = True)
    """
    # create left and right flames
    which_middle = flank // resolution
    if left:
        left_flames = stack[which_middle, which_middle + pad, :]

        return (left_flames)
    else:
        right_flames = stack[which_middle + pad, which_middle, :]
    return (right_flames)


def compute_fc_flames(stack, flank=10000, resolution=2000, remove_na=True, pad=3):
    assert pad >= 0, "Pad should be 0 or greater"

    # create left and right flames
    which_middle = flank // resolution
    # mask central pixel
    #     stack[which_middle, which_middle, :] = 0
    # mask 3*3 square
    #     stack[which_middle-1:which_middle+2, which_middle-1:which_middle+2, :] = 0

    if pad > 2:
        pad_lower = 2
    else:
        pad_lower = pad
    # compute right and left flames median
    right_flames = np.nanmedian(stack[which_middle - pad_lower:which_middle + pad + 1, which_middle, :], axis=0)
    left_flames = np.nanmedian(stack[which_middle, which_middle - pad:which_middle + pad_lower + 1, :], axis=0)
    assert right_flames.all() > 0  # division by 0 is problematic
    # fc = np.log10((left_flames + 0.01) / (right_flames + 0.01))
    fc = left_flames/right_flames
    return fc

def compute_flames(stack, flank=10000, resolution=2000, remove_na=True, pad=3):
    '''
    compute flames to be printed as separate columns
    '''
    assert pad >= 0, "Pad should be 0 or greater"

    # create left and right flames
    which_middle = flank // resolution
    # mask central pixel
    #     stack[which_middle, which_middle, :] = 0
    # mask 3*3 square
    #     stack[which_middle-1:which_middle+2, which_middle-1:which_middle+2, :] = 0

    if pad > 2:
        pad_lower = 2
    else:
        pad_lower = pad
    # compute right and left flames median
    right_flames = np.nanmedian(stack[which_middle - pad_lower:which_middle + pad + 1, which_middle, :], axis=0)
    left_flames = np.nanmedian(stack[which_middle, which_middle - pad:which_middle + pad_lower + 1, :], axis=0)
    assert right_flames.all() > 0  # division by 0 is problematic
    return left_flames.tolist(), right_flames.tolist()