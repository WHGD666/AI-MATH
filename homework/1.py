# 一维谐振子（有限差分法）数值求解 —— 完整可执行代码
# ====================== 1. 导入核心库 ======================
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

# ====================== 2. 设置网格参数 ======================
x_max = 5.0
N_int = 500
x = np.linspace(-x_max, x_max, N_int+2)
dx = x[1] - x[0]
x = x[1:-1]
N = len(x)

print(f"网格区间：[{x[0]:.2f}, {x[-1]:.2f}]，网格点数：{N}，间距：{dx:.4f}")

# ====================== 3. 构造哈密顿矩阵 ======================
kin = sparse.diags([1, -2, 1], offsets=[-1,0,1], shape=(N, N))
T = (-0.5 / dx**2) * kin
V_diag = 0.5 * x**2
V = sparse.diags(V_diag, offsets=0)
H_sparse = T + V
H_dense = H_sparse.toarray()

print(f"哈密顿矩阵形状：{H_dense.shape}")

# ====================== 4. 求解本征值问题 ======================
num_states = 6
evals_all, evecs_all = linalg.eigh(H_dense)
evals_sparse, evecs_sparse = eigsh(H_sparse, k=num_states, which='SM', tol=1e-8)
evals_sparse = np.sort(evals_sparse)

print("稠密对角化前6个能量：", evals_all[:num_states])
print("稀疏迭代求解前6个能量：", evals_sparse)

# ====================== 5. 提取并归一化波函数 ======================
E_num = evals_all[:num_states]
psi_num = evecs_all[:, :num_states]

psi_norm = np.zeros_like(psi_num)
for n in range(num_states):
    psi = psi_num[:, n]
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    psi_norm[:, n] = psi / norm

# ====================== 6. 验证奇偶性 ======================
print("\n波函数奇偶性验证：")
for n in range(2):
    psi = psi_norm[:, n]
    if np.allclose(psi, psi[::-1], atol=1e-6):
        parity = 'even'
    elif np.allclose(psi, -psi[::-1], atol=1e-6):
        parity = 'odd'
    else:
        parity = 'unknown'
    print(f"n={n}: energy={E_num[n]:.5f}, parity={parity}")

# ====================== 7. 绘制结果图 + 保存图片 ======================
plt.figure(figsize=(10, 6))
Vplot = 0.5 * x**2
plt.plot(x, Vplot, 'k-', linewidth=1.5, label='Potential V(x)')
colors = ['b','g','r','c','m','y']
for n in range(num_states):
    psi = psi_norm[:, n]
    E = E_num[n]
    plt.plot(x, psi * 0.8 + E, color=colors[n%len(colors)], label=f'n={n}')
    plt.hlines(E, x[0], x[-1], linestyles='--', colors='gray', alpha=0.7)
plt.ylim(-0.5, E_num[-1]+1)
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel(r'$\psi_n(x)$ (offset by $E_n$)')
plt.title('Quantum Harmonic Oscillator Eigenstates')
plt.grid(alpha=0.3)

# ✅ 新增：保存高清图片（PNG格式）
plt.savefig('谐振子波函数.png', dpi=300, bbox_inches='tight')

plt.show()

# ====================== 8. 结果保存与误差分析 ======================
# 保存数值数据（.npz格式，用于后续计算）
np.savez('sho_results.npz', E=E_num, x=x, psi=psi_norm)

E_analytic = np.array([n+0.5 for n in range(num_states)])
rel_error = np.abs((E_num - E_analytic) / E_analytic)

print("\n数值与解析能级 E_n=n+0.5 的相对误差：")
for n in range(num_states):
    print(f"n={n}: {rel_error[n]:.6f}")