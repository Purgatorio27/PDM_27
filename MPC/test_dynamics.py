#!/usr/bin/env python3
import numpy as np

# 验证动力学
x, y, v, psi = 0.0, 0.0, 1.0, np.radians(45)
delta, a = 0.0, 0.0
dt = 0.25
l_f, l_r = 0.75, 0.75

beta = np.arctan((l_r / (l_f + l_r)) * np.tan(delta))
print(f"beta = {np.degrees(beta):.2f} deg")

x_next = x + v * np.cos(psi + beta) * dt
y_next = y + v * np.sin(psi + beta) * dt
v_next = v + a * dt
psi_next = psi + (v / l_r) * np.sin(beta) * dt

print(f"Initial: x={x:.2f}, y={y:.2f}, v={v:.2f}, psi={np.degrees(psi):.1f} deg")
print(f"Next:    x={x_next:.2f}, y={y_next:.2f}, v={v_next:.2f}, psi={np.degrees(psi_next):.1f} deg")
print(f"Expected: x~0.177, y~0.177")
