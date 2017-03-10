import sys
from numpy import mat, mean, power


def read_input(file):
    for line in file:
        yield line.rstrip()


input = read_input(sys.stdin)
mapperout = [line.split('\t') for line in input]
cumval = 0.0
cumsumsq = 0.0
cumN = 0.0
for instance in mapperout:
    nj = float(instance[0])
    cumN += nj
    cumval += nj * float(instance[1])
    cumsumsq += nj * float(instance[2])
mean = cumval / cumN
varsum = (cumsumsq + cumN * mean * mean - 2.0 * mean * cumval) / cumN
print('%d\t%f\t%f' % (cumN, mean, varsum))
print('report: still alive', file=sys.stderr)
