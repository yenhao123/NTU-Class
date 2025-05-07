import math

def vc_bound(N, epsilon=0.1, delta=None, d_vc=3):
    term1 = 4 * (2 * N)**d_vc
    term2 = math.exp(-(1/8) * epsilon**2 * N)
    bound = term1 * term2
    return bound

# 範例：N = 100
print(vc_bound(100))        # ➜ 約 2.82e7
print(vc_bound(1000))       # ➜ 約 9.17e9
print(vc_bound(10000))      # ➜ 約 1.19e8
print(vc_bound(100000))     # ➜ 約 1.65e-38
print(vc_bound(29300))      # ➜ 約 9.99e-2 ≈ 0.1