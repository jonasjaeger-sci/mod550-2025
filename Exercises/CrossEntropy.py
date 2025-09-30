import numpy as np

true_dist_Px = np.array([0.3, 0.1, 0.4, 0.2])
model_dist_Qx = np.array([0.25, 0.25, 0.25, 0.25])

ind_cross_ent = true_dist_Px * np.log(1 / model_dist_Qx)
#ind_cross_ent = model_dist_Qx * np.log(1/model_dist_Qx)
#ind_cross_ent = true_dist_Px * np.log(1/true_dist_Px)
print(ind_cross_ent)
cross_ent = np.sum(ind_cross_ent)
print(f"The Cross-Entropy is: {cross_ent}")


