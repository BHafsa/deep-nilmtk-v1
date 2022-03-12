from deep_nilmtk.config import __models__

def present_other_framework(model_name):
    """
    Provide suggestions about other implementations if available
    """
    framework_name = None
    for framework in __models__:
        if model_name in __models__[framework].keys():
            framework_name = framework
            break
    return framework_name


def check_model_class(backend, model):
    """
    check if the custom model inherits from a propor class
    """
    base_class = __models__[backend][list(__models__[backend].keys())[0]]['model'].__bases__

    # if not issubclass(model, base_class):
    #     raise Exception(f'The custom model must inherit from class {base_class}')

def check_model_backend(backend, model, model_name):
    """
    check if the model is implemented in a proper backend
    """
    if model:
        check_model_class(backend, model)
    else:
        if model_name not in __models__[backend]:
            present = present_other_framework(model_name)
            if  present:
                msg = f'This  {model_name} is only available in {present} framework'
            else:
                msg = 'This model is not implemented in the toolkit'
            raise Exception(msg)

