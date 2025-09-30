import numpy as np

H_S = 1.571

car_own = np.array([0,1,1,0,1,0,1,1,2,2])
prob0 = 0.3
prob1 = 0.5
prob2 = 0.2

bus0 = 2/3
train0 = 1/3
car0 = 0

bus1 = 2/5
train1 = 2/5
car1 = 1/5

bus2= 0
train2 = 0
car2 = 1

H0 = -(bus0 * np.log2(bus0) + train0 * np.log2(train0))
H1 = -(bus1 * np.log2(bus1) + train1 * np.log2(train1) + car1 * np.log2(car1))
H2 = -(car2 * np.log2(car2))

print(H0)
print(H1)
print(H2)

InfoGain = 1.571 - prob0*H0- prob1*H1 - prob2*H2
print(f"The information Gain is: {InfoGain}")

