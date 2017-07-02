

'''python allows arguments to be passed like dictionaries by placing
"**" before the variable as it is referenced in the arguments to the
function. However, more than one of these cannot be employed, since
it is impossible to tell where the dictionary for the first ends and
the second dictionary begins. This class aims to mitigate that. It can
be referenced just like a dictionary using the __getitem__ python
special method. Many classes in this project will make use of this tool'''
class NamedArgs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getitem__(self, key):
        return self.kwargs[key]
