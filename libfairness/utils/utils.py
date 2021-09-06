

def column_group_merger(variable_groups):
    return [x[0].split("=")[0] for x in variable_groups]