import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro

# def call_r(script_path, inputs, output_var):
#     """
#     Runs an R script in Python and returns a specific variable from the R script as the return value.
#     :param script_path: path to the R script
#     :param inputs: dictionary of input variables
#     :param output_var: name of the variable to return from the R script
#     :return: value of the specified variable from the R script
#     """
#     r = robjects.r
#     # Load R script
#     with open(script_path, 'r') as f:
#         script = f.read()
#     # Convert pandas DataFrame to R DataFrame
#     pandas2ri.activate()
#     for key, value in inputs.items():
#         inputs[key] = pandas2ri.py2rpy(value)
#     # Send inputs to R
#     for key, value in inputs.items():
#         robjects.globalenv[key] = value
#     # Execute R script
#     r(script)
#     # Get the output variable from R
#     output_value = robjects.globalenv[output_var]
#
#     return output_value


def call_r(script_content, inputs, output_var):
    """
    Runs an R script in Python and returns a specific variable from the R script as the return value.
    :param script_content: content of the R script
    :param inputs: dictionary of input variables
    :param output_var: name of the variable to return from the R script
    :return: value of the specified variable from the R script
    """
    r = robjects.r
    # Convert pandas DataFrame to R DataFrame
    pandas2ri.activate()
    for key, value in inputs.items():
        inputs[key] = pandas2ri.py2rpy(value)
    # Send inputs to R
    for key, value in inputs.items():
        robjects.globalenv[key] = value
    # Execute R script
    r(script_content)
    # Get the output variable from R
    output_value = robjects.globalenv[output_var]

    return output_value
