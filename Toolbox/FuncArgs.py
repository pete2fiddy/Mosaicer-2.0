from Toolbox.NamedArgs import NamedArgs

'''often functions (so that they can be instantiated in the method)
must be passed to functions or other classes along with their argument.
It takes extra space and is confusing to pass the function and its
arguments (as a NamedArgs class) separately, so this class melds the two.
It is instantiated using the function and its arguments'''
class ClassArgs(NamedArgs):
    def __init__(self, func, **kwargs):
        NamedArgs.__init__(self, **kwargs)
        self.func = func
    
