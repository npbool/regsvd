import numpy as np

n_user_fea = 10
n_item_fea = 15
dim = 10
n_samples = 5000

ufmat = (np.random.rand(n_user_fea, dim)*2-1)*0.1
ifmat = (np.random.rand(n_item_fea, dim)*2-1)*0.1

ubias = (np.random.rand(n_user_fea)*2-1)*0.01
ibias = (np.random.rand(n_item_fea)*2-1)*0.01

print(np.sum(ufmat))
print(ufmat[0])
print(np.max(ufmat))
print(np.sum(ubias))
bias = 0
def gen_sample():
    while 1:
        ufea = np.random.rand(n_user_fea)>0.7
        ifea = np.random.rand(n_item_fea)>0.7
        if np.sum(ufea)>0 and np.sum(ifea)>0:
            return (ufea, ifea)

def calc_label(sample):
    x = 0
    ufv = np.sum((ufmat[i] for i in sample[0]))
    ifv = np.sum((ifmat[i] for i in sample[1]))

    x += ufv.dot(ifv)

    x += np.sum((ubias[i] for i in sample[0]))
    x += np.sum((ibias[i] for i in sample[1]))

    if x>=0:
        return 1
    else:
        return -1

if __name__=="__main__":
    samples = [ gen_sample() for i in range(n_samples) ]
    labels = [calc_label(sample) for sample in samples]

    of = open("data", "w")
    of.write("%d %d %d\n" % (n_user_fea, n_item_fea, n_samples))
    for s,l in zip(samples, labels):
        of.write("%d\t" % l)
        of.write("%d %d\t" % (np.sum(s[0]), np.sum(s[1])))

        for i in range(n_user_fea):
            if s[0][i]:
                of.write(str(i)+" ")
        for i in range(n_item_fea):
            if s[1][i]:
                of.write(str(i)+" ")
        of.write("\n")
    of.close()
