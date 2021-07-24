import numpy as np

mu = 0.9
user_number = 10
G = np.zeros((user_number, user_number), dtype=np.float)
for i in range(user_number):
    for j in range(user_number):
        if i == j:
            G[i][j] = 0
        else:
            p = np.random.random()
            if p <= mu:
                G[i][j] = 1
            else:
                G[i][j] = 0
for i in range(user_number):
    print('[', end='')
    for j in range(user_number):
        print(G[i][j], end=', ')
    print('], ')
# G = np.array([
#     [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, ],
#     [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, ],
#     [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
#     [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, ],
#     [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
#     [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, ],
#     [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, ],
#     [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, ],
#     [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, ],
#     [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, ],
# ])

I = np.ones((user_number, 1))
K = np.dot(G, I)
L = np.dot(G.transpose(), I)
g = np.sum(G)
d = g / user_number
H = np.dot(K, L.transpose()) / g
for i in range(user_number):
    print('[', end='')
    for j in range(user_number):
        # if i == j:
        #     print('0.0', end=', ')
        # else:
        print(np.round(H[i][j], 2), end=', ')
    print('],')