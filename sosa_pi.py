#Dylan Sosa
#31 August 2016
import random, math
def monte_carlo_pi(num_darts):
    m = 0
    n = 0
    for n in range(num_darts):
        x = random.random()
        y = random.random ()
        d = math.sqrt(x**2 + y**2)
        if d < 1:
            m = m + 1
        else:
            m = m
    n = n + 1
    pi = (4.0 * m) / n
    print "Approximation:",pi
    return pi

monte_carlo_pi(1000000) #getting closer!
