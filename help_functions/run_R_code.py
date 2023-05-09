import rpy2.robjects as robjects

def run_r_script(script_path, inputs):
    """
    The function runs R scripts in Python and returns the output of the R code.
    :param script_path: path to the R script
    :param inputs: df with the log returns for each fund.
    :return: output created in the R script is returned.
    """
    r = robjects.r
    # load R script
    with open(script_path, 'r') as f:
        script = f.read()
    # send inputs to R
    for key, value in inputs.items():
        r[key] = value
    # execute R script
    r(script)
    # get outputs from R
    outputs = {}
    for key in r.keys():
        if key != ".GlobalEnv":
            outputs[key] = r[key]
    return outputs
