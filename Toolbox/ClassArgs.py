from Toolbox.NamedArgs import NamedArgs

'''often class types (so that they can be instantiated in the method)
must be passed to functions or other classes along with their argument.
It takes extra space and is confusing to pass the class type and its
arguments (as a NamedArgs class) separately, so this class melds the two.
It is instantiated using the class type and its arguments, and provides
functionality to initialize the class and return the instance'''
class ClassArgs(NamedArgs):
    def __init__(self, class_type, **kwargs):
        NamedArgs.__init__(self, **kwargs)
        self.class_type = class_type
