

def get_variable_name(ref_obj, outer_locals):
    '''
    Example usage:
        blubb = 42
        var_str = get_variable_name(blubb, locals())
        print(var_str, blubb)
        # blubb 42
    :param ref_obj:
    :param outer_locals:
    :return:
    '''
    variable_name = [i for i, j in outer_locals.items() if j == ref_obj][0]
    return variable_name

