from utils.datareader import Datareader
import scipy.sparse as sps

old = sps.load_npz("online-test-old.npz")
new = sps.load_npz("online-test-new.npz")
dr_old = Datareader(verbose=False, mode='online', only_load=True, type="old")
dr_new = Datareader(verbose=False, mode='online', only_load=True)


#### controllo pids ####

for i in range(1,11):
    print("indices new cat"+str(i)+": "+str(dr_new.get_test_pids_indices(cat=i)))
    print("indices old cat"+str(i)+": "+str(dr_old.get_test_pids_indices(cat=i)))


#### Controllo correttezza split ####

for i in range(1,11):
    indices_old = dr_old.get_test_pids_indices(cat=i)
    indices_new = dr_new.get_test_pids_indices(cat=i)
    print(indices_old)
    res = old[indices_old] - new[indices_new]
    assert res.nnz == 0
    print("Split cat"+str(i)+": ok")


#### Controllo operazioni di ensembling after split

res = []
for i in range(1,11):
    indices_old = dr_old.get_test_pids_indices(cat=i)
    res.append(old[indices_old])
old_vstack = sps.vstack(res)
res = []
for i in range(1,11):
    indices_new = dr_new.get_test_pids_indices(cat=i)
    res.append(new[indices_new])
new_vstack = sps.vstack(res)
res = old_vstack - new_vstack
assert res.nnz == 0
print("Riensemble after splitting ok")