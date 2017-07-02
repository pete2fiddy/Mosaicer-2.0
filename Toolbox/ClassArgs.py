from Toolbox.NamedArgs import NamedArgs

'''often class types (so that they can be instantiated in the method)
must be passed to functions or other classes along with their argument.
It takes extra space and is confusing to pass the class type and its
arguments (as a NamedArgs class) separately, so this class melds the two.
It is instantiated using the class type and its arguments, and provides
functionality to initialize the class and return the instance'''
class ClassArgs:
    def __init__(self, class_type, **kwargs):
        self.class_type = class_type
        self.kwargs = NamedArgs(**kwargs)

    '''write an "init" function that either uses only the named args
    to init the class and return it, or somehow (not sure how yet)
    inits the class type using additional arguments as well as
    the named args stored in this instance'''

    def __getitem__(self, key):
        return self.kwargs[key]
