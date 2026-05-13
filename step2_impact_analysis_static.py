# 10주차 실습: 충격 특성 분석 및 손상 예측 모델링
# 가상 Tracker 속도값을 활용한 정적 분석 코드

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

g = 9.81
mass = 0.25
drop_height = 1.0
t_total = 1.2

# 가상 Tracker 속도 데이터 기반 반발 계수
e_hard = 1.50 / 4.40
e_soft = 0.80 / 4.40

dt_hard = 0.005
dt_soft = 0.020
damage_threshold = 150

def calculate_impact(e, dt):
    t_fall = np.sqrt(2 * drop_height / g)
    v1 = -np.sqrt(2 * g * drop_height)
    v2 = abs(v1) * e
    impact_force = mass * (v2 - v1) / dt

    time = np.linspace(0, t_total, 3000)
    y_points, force_points = [], []

    for t in time:
        if t < t_fall:
            y = drop_height - 0.5 * g * t**2
            force = 0
        elif t < t_fall + dt:
            y = 0
            force = impact_force
        else:
            t_b = t - (t_fall + dt)
            y = v2 * t_b - 0.5 * g * t_b**2
            y = max(y, 0)
            force = 0

        y_points.append(y)
        force_points.append(force)

    return time, np.array(y_points), np.array(force_points), t_fall, v1, v2, impact_force

t, y_h, f_h, tf, v1_h, v2_h, F_h = calculate_impact(e_hard, dt_hard)
_, y_s, f_s, _, v1_s, v2_s, F_s = calculate_impact(e_soft, dt_soft)

print('===== 충격 특성 분석 결과 =====')
print(f'맨바닥 반발계수: {e_hard:.2f}')
print(f'완충재 반발계수: {e_soft:.2f}')
print(f'맨바닥 최대 충격력: {F_h:.1f} N')
print(f'완충재 최대 충격력: {F_s:.1f} N')
print(f'충격력 감소율: {(1 - F_s / F_h) * 100:.1f} %')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

ax1.plot(t, y_h, 'r', lw=2, label='맨바닥')
ax1.plot(t, y_s, 'g', lw=2, label='완충재')
ax1.set_title('과일 낙하 높이-시간 그래프')
ax1.set_xlabel('시간 (s)')
ax1.set_ylabel('높이 (m)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(t, f_h, 'r', lw=2, label=f'맨바닥: {F_h:.1f} N')
ax2.plot(t, f_s, 'g', lw=2, label=f'완충재: {F_s:.1f} N')
ax2.axhline(damage_threshold, color='orange', linestyle='--', lw=2, label=f'손상 임계치 {damage_threshold} N')

axins = inset_axes(ax2, width='35%', height='45%', loc='upper right', borderpad=2)
axins.plot(t, f_h, 'r')
axins.plot(t, f_s, 'g')
axins.axhline(damage_threshold, color='orange', linestyle='--', lw=1)
axins.set_xlim(tf - 0.01, tf + 0.04)
axins.set_ylim(0, max(F_h, F_s) * 1.1)
axins.grid(True, alpha=0.2)

ax2.set_title('충격력-시간 그래프 및 손상 임계치 비교')
ax2.set_xlabel('시간 (s)')
ax2.set_ylabel('충격력 (N)')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('step1_impact_analysis_result.png', dpi=300, bbox_inches='tight')
plt.show()
