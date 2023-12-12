from sklearn.neighbors import BallTree
import numpy as np
import time

def is_linear_or_log(X):
	diffs = np.diff(X)
	ratios = np.divide(diffs[:-1], diffs[1:])
	avg_diff = np.mean(diffs)
	avg_ratio = np.mean(ratios)
	std_diff = np.std(diffs)
	std_ratio = np.std(ratios)
	if abs(std_diff) < 1e-4:
		return "linear"
	elif abs(std_ratio) < 1e-4:
		return "log"
	else:
		return "neither"
    
    
def get_theta_tree(data, query, k):
	start = time.time()
	# the ball tree.
	xtree = BallTree(data, metric='haversine') # dec, ra
	print("build tree", time.time()-start)

	start = time.time()
	dis_theta = xtree.query(query, k=k)
	print("query", time.time()-start)

	return dis_theta

def calc_cdf_hist_theta(thetas, dis_theta):
    
	cdfs = np.zeros((dis_theta.shape[1], len(thetas)), dtype = np.float32)
    
	# are the bins lin or log
	thetabins = np.concatenate((np.zeros(1, dtype = np.float32), thetas))
	scaling_t = is_linear_or_log(thetabins)
    
	start = time.time()
	print(dis_theta.shape, thetabins)
	for ik in range(dis_theta.shape[1]):
		dist_hist_k, _ = np.histogram(dis_theta[:, ik], bins=thetabins) 
		dist_cdf_k = np.cumsum(dist_hist_k)
		cdfs[ik] = dist_cdf_k / dist_cdf_k[-1]
    
	return cdfs


def CDFkNN_theta(thetas, xgal, xrand, kneighbors = 1, nthread = 32, randdown = 1): # xgal and xrand in (dec, ra) in radians, theta also in radians
    
	assert xgal.shape[1] == 2
	assert np.max(xgal[:, 0]) <= np.pi/2
	assert xrand.shape[1] == 2
    
	thetas = np.float32(thetas)
	xgal = np.float32(xgal)
	xrand = np.float32(xrand)
    
	xgal = np.array(xgal, order="C")
	xrand = np.array(xrand, order="C")
    
	if randdown > 1:
		xrand = xrand[np.random.choice(np.arange(len(xrand)), size = int(len(xrand)/randdown), replace = False)]
	print("Ngal", len(xgal), "Nrand", len(xrand), kneighbors)
    
	start = time.time()

	dis_theta = get_theta_tree(xgal, xrand, kneighbors)[0]
	# print(dis_theta[0].shape)

	print("  kdtree tot", time.time() - start) 

	start = time.time()
	outputs1 = calc_cdf_hist_theta(thetas, dis_theta)
	print("  cdf", time.time() - start)

	return outputs1
