import numpy as np
# import matplotlib.pyplot as plt


def customer_split(customers: np.ndarray, locker_idx: np.ndarray, loc: int):
    split_cus = list()
    for idx in range(loc):
        split_cus.append(customers[np.where(locker_idx == idx)])

    return split_cus


def locker_region(lockers: np.ndarray, customers: np.ndarray):
    tri_cus = np.expand_dims(customers, 0).repeat(len(lockers), 0)
    tri_cus = np.transpose(tri_cus, [1, 0, 2])
    mul_loc = np.expand_dims(lockers, 0).repeat(len(customers), 0)
    dif = mul_loc - tri_cus
    ds = np.linalg.norm(dif, axis=2)
    min_idx = np.argmin(ds, axis=1)
    return min_idx


def data_generate(loc=3, cus=15):
    mean = (0, 0)
    cov = [[10000, 5000], [5000, 10000]]
    lockers = np.random.multivariate_normal(mean, cov, loc).round(2)
    # print(lockers)

    # plt.figure()
    # l_c = ['r', 'g', 'b']
    # c_c = ["tomato", "lime", "navy"]
    all_customer = np.array([[0, 0]])
    for idx, locker in enumerate(lockers):
        mean = locker
        cov = [[1000, 500], [500, 1000]]
        customers = np.random.multivariate_normal(mean, cov, cus).round(2)
        all_customer = np.concatenate((all_customer, customers), axis=0)

        # print(customers)
        # plt.scatter(locker[0], locker[1], marker='o', c=l_c[idx])
        # plt.scatter(customers.transpose()[0], customers.transpose()[1], marker='*', c=c_c[idx])

    # plt.show()

    all_customer = all_customer[1:]
    locker_idx = locker_region(lockers, all_customer)

    split_cus = customer_split(all_customer, locker_idx, loc)

    return lockers, all_customer, split_cus, locker_idx


if __name__ == '__main__':
    data_generate()
