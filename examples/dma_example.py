import numpy as np
import matplotlib.pyplot as plt
import fathon
from fathon import fathonUtils as fu

print('This is fathon v{}'.format(fathon.__version__))

a = np.random.randn(10000)
b = np.random.randn(10000)

a = fu.toAggregated(a)
b = fu.toAggregated(b)

pydma = fathon.DMA(a)

winSizes = fu.linRangeByStep(10, 2000)
revSeg = True
polOrd = 3

n, F = pydma.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)

H, H_intercept = pydma.fitFlucVec()

plt.plot(np.log(n), np.log(F), 'ro')
plt.plot(np.log(n), H_intercept+H*np.log(n), 'k-', label='H = {:.2f}'.format(H))
plt.xlabel('ln(n)', fontsize=14)
plt.ylabel('ln(F(n))', fontsize=14)
plt.title('DMA', fontsize=14)
plt.legend(loc=0, fontsize=14)
plt.show()

limits_list = np.array([[15,2000], [200,1000]], dtype=int)
list_H, list_H_intercept = pydma.multiFitFlucVec(limits_list)

clrs = ['k', 'b', 'm', 'c', 'y']
stls = ['-', '--', '.-']
plt.plot(np.log(n), np.log(F), 'ro')
for i in range(len(list_H)):
    n_rng = np.arange(limits_list[i][0], limits_list[i][1]+1)
    plt.plot(np.log(n_rng), list_H_intercept[i]+list_H[i]*np.log(n_rng),
             clrs[i%len(clrs)]+stls[(i//len(clrs))%len(stls)], label='H = {:.2f}'.format(list_H[i]))
plt.xlabel('ln(n)', fontsize=14)
plt.ylabel('ln(F(n))', fontsize=14)
plt.title('DMA', fontsize=14)
plt.legend(loc=0, fontsize=14)
plt.show()
