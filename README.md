Physics-Informed Neural Networks (PINNs) Hyperparameter Study

Comprehensive evaluation of different architectures and activation functions for solving the 1D diffusion equation:

∂y/∂t = ∂²y/∂x² - e^(-t)(sin(πx) - π²sin(πx))


Domain: x ∈ [-1,1], t ∈ [0,1]
Initial Condition: y(x,0) = sin(πx)
Boundary Conditions: y(-1,t) = 0, y(1,t) = 0
Exact solution: y(x,t) = e^(-t)sin(πx)

<img width="1380" height="671" alt="image" src="https://github.com/user-attachments/assets/a6d153e4-4103-4530-a687-4859b88981e5" />
