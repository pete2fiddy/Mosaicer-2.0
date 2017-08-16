

class UnivariateGaussian:

    def __init__(self, mean, var):
        self.mean = mu
        self.var = var

    '''returns the normal distribution created by convolving this gaussian
    with the other gaussian'''
    def convolve(self, other_gauss):
        new_mu = self.mean + other_gauss.mean
        new_var = self.var + other_gauss.var
        return UnivariateGaussian(new_mu, new_var)

    '''returns the normalized product between this gaussian and the other,
    normalized as in it integrates to one.
    Note the product of two distributions is equivalent to the joint probability
    of both (in this case the normalized one)'''
    def norm_product(self, other_gauss):
        mean2 = other_gauss.mean
        var2 = other_gauss.var
        new_mu = (mean2*self.var + self.mean*var2)/(self.var + var2)
        new_var = (self.var*var2)/(self.var+var2)
        return UnivariateGaussian(new_mu, new_var)
