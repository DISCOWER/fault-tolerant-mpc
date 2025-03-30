# Inertial Properties of 2D Freeflyer and 3D Spacecraft Examples

| Robot:              | 2D Freeflyer | 3D Spacecraft |
| ---------------- | ------ | ---- |
| $\delta_t$ [s]       |   0.1   | 0.1 |
| Mass $m$ [kg]          |   14.5   | 16.8 |
| Inertia $J$ [kg $m^2$]   |  0.370   | diag($0.2,0.3,0.25$) |
| Maximum force $f^{max}$ [N] |  1.75   | 1.75 |

## Allocation Matrices $D$

### 2D Freeflyer

$$
\begin{align*}
    D = \left[ \begin{smallmatrix}
        1& -1&  1& -1&  0&  0&  0&  0\\
        0&  0&  0&  0&  1& -1&  1& -1\\
        d& -d& -d&  d&  d& -d& -d&  d
    \end{smallmatrix} \right]
\end{align*}
$$

### 3D Spacecraft

$$
\begin{align*}
     D = 
     \begin{bmatrix}
-1&-1&1&1&-1&-1&1&1&0&0&0&0&0&0&0&0\\0&0&0&0&0&0&0&0&-1&-1&1&1&0&0&0&0\\0&0&0&0&0&0&0&0&0&0&0&0&-1&1&-1&1\\0&0&0&0&0&0&0&0&0&0&0&0&-a&a&a&-a\\-c&c&c&-c&-c&c&c&-c&0&0&0&0&0&0&0&0\\a&a&-a&-a&-a&-a&a&a&-b&b&b&-b&0&0&0&0
     \end{bmatrix}
\end{align*}
$$
with a=12 cm, b=9 cm and c=5 cm.