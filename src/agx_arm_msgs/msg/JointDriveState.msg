# 한 관절의 모터(high-speed) + 드라이버(low-speed) 상태.
# get_motor_states(joint) + get_driver_states(joint) 결과를 함께 담는다.
string joint_name

# --- get_motor_states (high-speed feedback, CAN 0x251~0x257) ---
float64 velocity            # 모터 속도 (rad/s)
float64 current             # 모터 전류 (A)
float64 position            # 모터 위치 (rad, motor side)
float64 torque              # 관절측 토크 (N.m, 전류·감속비 환산)

# --- get_driver_states (low-speed feedback, CAN 0x261~0x267) ---
float64 voltage             # 드라이버 버스 전압 (V)
float64 driver_temp         # 드라이버(FOC) 온도 (degC)
float64 motor_temp          # 모터 온도 (degC)
float64 bus_current         # 버스 전류 (A); 1.5kg 관절은 미샘플 → 0
uint8 foc_status_code       # 드라이버 상태 raw 바이트 (아래 bool로 비트 분해)
bool voltage_too_low        # bit0: 전원전압 과저
bool motor_overheating      # bit1: 모터 과온
bool driver_overcurrent     # bit2: 드라이버 과전류
bool driver_overheating     # bit3: 드라이버 과온
bool collision_status       # bit4: 충돌보호 트리거
bool driver_error_status    # bit5: 드라이버 에러
bool driver_enable_status   # bit6: 드라이버 enable
bool stall_status           # bit7: 스톨(堵转)보호 트리거
