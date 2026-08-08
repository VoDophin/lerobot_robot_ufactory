from lerobot.motors import Motor, MotorCalibration, MotorNormMode


MOTORS = {
    "joint1": Motor(1, "AGILEX-M", MotorNormMode.RANGE_M100_100),
    "joint2": Motor(2, "AGILEX-M", MotorNormMode.RANGE_M100_100),
    "joint3": Motor(3, "AGILEX-M", MotorNormMode.RANGE_M100_100),
    "joint4": Motor(4, "AGILEX-S", MotorNormMode.RANGE_M100_100),
    "joint5": Motor(5, "AGILEX-S", MotorNormMode.RANGE_M100_100),
    "joint6": Motor(6, "AGILEX-S", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(7, "AGILEX-S", MotorNormMode.RANGE_0_100),
}

CALIBRATION = {
    "joint1": MotorCalibration(1, 0, 0, -150000, 150000),
    "joint2": MotorCalibration(2, 0, 0, 0, 180000),
    "joint3": MotorCalibration(3, 0, 0, -170000, 0),
    "joint4": MotorCalibration(4, 0, 0, -100000, 100000),
    "joint5": MotorCalibration(5, 0, 0, -65000, 65000),
    "joint6": MotorCalibration(6, 0, 0, -100000, 130000),
    "gripper": MotorCalibration(7, 0, 0, 0, 68000),
}

PARKING_POSITION = {
    "joint1": 0.0,
    "joint2": -100.0,
    "joint3": 100.0,
    "joint4": 0.0,
    "joint5": 35.0,
    "joint6": 0.0,
    "gripper": 0.0,
}

