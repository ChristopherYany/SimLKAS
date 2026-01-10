from dataclasses import dataclass, fields
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from rotations import angle_normalize

LaneParams = Dict[str, Optional[Tuple[float, float, float, float]]]


@dataclass
class FusionConfig:
    # Image and lane geometry
    image_width_px: float = 640.0
    image_height_px: float = 480.0
    lane_width_px: float = 420.0
    lane_width_m: float = 3.5
    lane_center_px: float = 320.0
    lookahead_px_base: float = 700.0
    lookahead_px_speed_gain: float = -5.0
    lookahead_px_min: float = 120.0
    lookahead_px_max: float = 470.0

    # Process and measurement noise
    imu_accel_noise: float = 0.5
    imu_gyro_noise: float = 0.02
    gnss_speed_noise: float = 0.5
    lane_d_noise: float = 0.25
    lane_psi_noise: float = 0.08

    # Adaptive lane measurement noise scaling
    lane_r_scale_min: float = 0.5
    lane_r_scale_max: float = 8.0
    conf_alpha: float = 0.8
    conf_low_th: float = 0.3
    conf_high_th: float = 0.6
    width_tol_px: float = 120.0
    angle_tol_rad: float = 0.35

    # Smoothing and gating
    output_tau: float = 0.4
    residual_clip_d: float = 1.0
    residual_clip_psi: float = 0.4
    nis_gate: float = 9.21  # chi2(2 dof, 0.99)

    # Feature toggles
    enable_adaptive: bool = True
    enable_nis_gate: bool = True
    enable_output_smoothing: bool = True
    use_gnss_speed: bool = True

    # Sign and bias adjustments (CARLA defaults usually work with yaw_rate_sign=-1)
    yaw_rate_sign: float = -1.0
    accel_x_sign: float = 1.0
    imu_accel_bias_x: float = 0.0
    imu_gyro_bias_z: float = 0.0

    # Speed bounds and initialization covariance
    min_speed_mps: float = 0.0
    max_speed_mps: float = 60.0
    init_cov_d: float = 0.6
    init_cov_psi: float = 0.4
    init_cov_v: float = 5.0

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "FusionConfig":
        if not data:
            return cls()
        allowed = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in allowed}
        return cls(**kwargs)


class LaneFusionEKF:
    """
    LKS-oriented EKF that fuses lane measurements (vision) with IMU and GNSS speed.

    State x = [d, psi, v]
      d   : lateral offset (m), positive when lane center is to the right of vehicle
      psi : heading error (rad), lane direction minus vehicle heading
      v   : vehicle speed (m/s)
    """

    def __init__(self, config: FusionConfig):
        self._cfg = config
        self._x = np.zeros(3)
        self._P = np.diag([
            config.init_cov_d ** 2,
            config.init_cov_psi ** 2,
            config.init_cov_v ** 2,
        ])
        self._x_out = self._x.copy()

        self._time: Optional[float] = None
        self._last_output_time: Optional[float] = None
        self._last_imu: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._target_speed_kph: Optional[float] = None

        self._imu_queue: Deque[Tuple[float, np.ndarray, np.ndarray]] = deque()
        self._gnss_queue: Deque[Tuple[float, float, float, float]] = deque()
        self._lane_queue: Deque[Tuple[float, LaneParams]] = deque()

        self._initialized = False
        self._pending_speed_mps: Optional[float] = None
        self._last_lane_raw: Optional[LaneParams] = None
        self._lane_params_out: Optional[LaneParams] = None

        self._conf_smooth = 1.0
        self._vision_ok = True

        self._gnss_origin: Optional[Tuple[float, float]] = None
        self._last_gnss_pos: Optional[Tuple[float, float]] = None
        self._last_gnss_t: Optional[float] = None

    def set_target_speed_kph(self, target_speed_kph: float) -> None:
        self._target_speed_kph = float(target_speed_kph)

    def push_imu(self, t: float, accel: np.ndarray, gyro: np.ndarray) -> None:
        if t is None:
            return
        if self._imu_queue and t < self._imu_queue[-1][0]:
            return
        self._imu_queue.append((float(t), np.asarray(accel, dtype=float), np.asarray(gyro, dtype=float)))

    def push_gnss(self, t: float, lat: float, lon: float, alt: float) -> None:
        if t is None:
            return
        if self._gnss_queue and t < self._gnss_queue[-1][0]:
            return
        self._gnss_queue.append((float(t), float(lat), float(lon), float(alt)))

    def push_lane(self, t: float, lane_params: LaneParams, target_speed_kph: Optional[float] = None) -> None:
        if t is None or lane_params is None:
            return
        if target_speed_kph is not None:
            self.set_target_speed_kph(target_speed_kph)
        if self._lane_queue and t < self._lane_queue[-1][0]:
            return
        self._last_lane_raw = lane_params
        self._lane_queue.append((float(t), lane_params))

    def advance(self, t_target: float) -> None:
        if t_target is None:
            return
        if self._time is None:
            self._time = float(t_target)
            self._last_output_time = self._time
            return

        while True:
            t_next, kind = self._peek_next_event()
            if t_next is None or t_next > t_target:
                break

            self._predict_to(t_next)

            if kind == "imu":
                t, accel, gyro = self._imu_queue.popleft()
                self._last_imu = (accel, gyro)
                self._time = t
            elif kind == "gnss":
                t, lat, lon, alt = self._gnss_queue.popleft()
                self._time = t
                self._handle_gnss(t, lat, lon, alt)
            elif kind == "lane":
                t, lane_params = self._lane_queue.popleft()
                self._time = t
                self._handle_lane(t, lane_params)

        self._predict_to(float(t_target))
        self._time = float(t_target)
        self._update_output(self._time)

    def get_lane_params(self) -> Optional[LaneParams]:
        if self._lane_params_out is not None:
            return self._lane_params_out
        return self._last_lane_raw

    def get_debug_state(self) -> dict:
        return {
            "time": self._time,
            "state": self._x.copy(),
            "state_out": self._x_out.copy(),
            "confidence": self._conf_smooth,
            "vision_ok": self._vision_ok,
        }

    def _peek_next_event(self) -> Tuple[Optional[float], Optional[str]]:
        candidates = []
        if self._imu_queue:
            candidates.append((self._imu_queue[0][0], "imu"))
        if self._gnss_queue:
            candidates.append((self._gnss_queue[0][0], "gnss"))
        if self._lane_queue:
            candidates.append((self._lane_queue[0][0], "lane"))
        if not candidates:
            return None, None
        return min(candidates, key=lambda item: item[0])

    def _predict_to(self, t_next: float) -> None:
        if self._last_imu is None or self._time is None:
            return
        dt = float(t_next) - float(self._time)
        if dt <= 0.0:
            return

        accel, gyro = self._last_imu
        a_long = self._cfg.accel_x_sign * (accel[0] - self._cfg.imu_accel_bias_x)
        yaw_rate = self._cfg.yaw_rate_sign * (gyro[2] - self._cfg.imu_gyro_bias_z)

        d, psi, v = self._x
        d_dot = v * np.sin(psi)
        psi_dot = yaw_rate
        v_dot = a_long

        self._x = self._x + dt * np.array([d_dot, psi_dot, v_dot])
        self._x[1] = angle_normalize(np.array([self._x[1]]))[0]
        self._x[2] = float(np.clip(self._x[2], self._cfg.min_speed_mps, self._cfg.max_speed_mps))

        F = np.eye(3)
        F[0, 1] = dt * v * np.cos(psi)
        F[0, 2] = dt * np.sin(psi)

        q_d = (0.5 * dt ** 2 * self._cfg.imu_accel_noise) ** 2
        q_psi = (dt * self._cfg.imu_gyro_noise) ** 2
        q_v = (dt * self._cfg.imu_accel_noise) ** 2
        Q = np.diag([q_d, q_psi, q_v])

        self._P = F @ self._P @ F.T + Q

    def _handle_gnss(self, t: float, lat: float, lon: float, alt: float) -> None:
        if not self._cfg.use_gnss_speed:
            return
        if self._gnss_origin is None:
            self._gnss_origin = (lat, lon)
            self._last_gnss_pos = (0.0, 0.0)
            self._last_gnss_t = t
            return

        x, y = self._geo_to_local(lat, lon)
        if self._last_gnss_pos is not None and self._last_gnss_t is not None:
            dt = t - self._last_gnss_t
            if dt > 0.0:
                dx = x - self._last_gnss_pos[0]
                dy = y - self._last_gnss_pos[1]
                speed = np.hypot(dx, dy) / dt
                self._update_speed(speed)

        self._last_gnss_pos = (x, y)
        self._last_gnss_t = t

    def _handle_lane(self, t: float, lane_params: LaneParams) -> None:
        meas = self._extract_lane_measurement(lane_params)
        if meas is None:
            return

        d_meas, psi_meas, conf = meas
        self._update_confidence(conf)

        if not self._initialized:
            v0 = self._pending_speed_mps
            if v0 is None:
                v0 = self._target_speed_to_mps()
            self._x = np.array([d_meas, psi_meas, float(v0)])
            self._x_out = self._x.copy()
            self._initialized = True
            return

        H = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]])
        z = np.array([d_meas, psi_meas])
        y = z - H @ self._x
        y[1] = angle_normalize(np.array([y[1]]))[0]
        y[0] = float(np.clip(y[0], -self._cfg.residual_clip_d, self._cfg.residual_clip_d))
        y[1] = float(np.clip(y[1], -self._cfg.residual_clip_psi, self._cfg.residual_clip_psi))

        R_base = np.diag([self._cfg.lane_d_noise ** 2, self._cfg.lane_psi_noise ** 2])
        scale = self._lane_r_scale()
        R = R_base * scale
        S = H @ self._P @ H.T + R

        if self._cfg.enable_nis_gate:
            try:
                nis = float(y.T @ np.linalg.inv(S) @ y)
            except np.linalg.LinAlgError:
                nis = 0.0
            if nis > self._cfg.nis_gate:
                if self._cfg.enable_adaptive:
                    scale = max(scale, self._cfg.lane_r_scale_max)
                    R = R_base * scale
                    S = H @ self._P @ H.T + R
                else:
                    return

        try:
            K = self._P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        self._x = self._x + K @ y
        self._x[1] = angle_normalize(np.array([self._x[1]]))[0]
        self._x[2] = float(np.clip(self._x[2], self._cfg.min_speed_mps, self._cfg.max_speed_mps))
        I = np.eye(3)
        self._P = (I - K @ H) @ self._P

    def _update_speed(self, speed_mps: float) -> None:
        speed = float(np.clip(speed_mps, self._cfg.min_speed_mps, self._cfg.max_speed_mps))
        if not self._initialized:
            self._pending_speed_mps = speed
            return

        H = np.array([[0.0, 0.0, 1.0]])
        z = np.array([speed])
        y = z - H @ self._x
        R = np.array([[self._cfg.gnss_speed_noise ** 2]])
        S = H @ self._P @ H.T + R
        try:
            K = self._P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        self._x = self._x + (K @ y).reshape(-1)
        self._x[2] = float(np.clip(self._x[2], self._cfg.min_speed_mps, self._cfg.max_speed_mps))
        I = np.eye(3)
        self._P = (I - K @ H) @ self._P

    def _update_confidence(self, conf_raw: float) -> None:
        conf = float(np.clip(conf_raw, 0.0, 1.0))
        if self._cfg.enable_adaptive:
            self._conf_smooth = (
                self._cfg.conf_alpha * self._conf_smooth + (1.0 - self._cfg.conf_alpha) * conf
            )
            if self._vision_ok and conf < self._cfg.conf_low_th:
                self._vision_ok = False
            elif (not self._vision_ok) and conf > self._cfg.conf_high_th:
                self._vision_ok = True
        else:
            self._conf_smooth = conf
            self._vision_ok = True

    def _lane_r_scale(self) -> float:
        if not self._cfg.enable_adaptive:
            return 1.0
        conf = self._conf_smooth if self._vision_ok else min(self._conf_smooth, self._cfg.conf_low_th)
        scale = self._cfg.lane_r_scale_min + (1.0 - conf) * (
            self._cfg.lane_r_scale_max - self._cfg.lane_r_scale_min
        )
        if not self._vision_ok:
            scale = max(scale, self._cfg.lane_r_scale_max)
        return float(np.clip(scale, self._cfg.lane_r_scale_min, self._cfg.lane_r_scale_max))

    def _extract_lane_measurement(self, lane_params: LaneParams) -> Optional[Tuple[float, float, float]]:
        if lane_params is None:
            return None
        left = lane_params.get("left")
        right = lane_params.get("right")
        y_ref = self._lookahead_px()

        left_x = self._line_x_at_y(left, y_ref)
        right_x = self._line_x_at_y(right, y_ref)
        left_angle = self._line_angle(left)
        right_angle = self._line_angle(right)

        left_valid = left_x is not None and left_angle is not None
        right_valid = right_x is not None and right_angle is not None

        if not left_valid and not right_valid:
            return None

        if left_valid and right_valid:
            if right_x <= left_x:
                return None
            center_x = (left_x + right_x) / 2.0
        elif left_valid:
            center_x = left_x + self._cfg.lane_width_px / 2.0
        else:
            center_x = right_x - self._cfg.lane_width_px / 2.0

        pixel_to_meter = self._cfg.lane_width_m / self._cfg.lane_width_px
        d_m = (center_x - self._cfg.lane_center_px) * pixel_to_meter

        if left_valid and right_valid:
            lane_angle = angle_normalize(np.array([(left_angle + right_angle) / 2.0]))[0]
        else:
            lane_angle = left_angle if left_valid else right_angle
        psi = angle_normalize(np.array([lane_angle - np.pi / 2.0]))[0]

        conf = 1.0 if (left_valid and right_valid) else 0.6
        if left_valid and right_valid:
            width_px = right_x - left_x
            width_err = abs(width_px - self._cfg.lane_width_px)
            width_score = max(0.0, 1.0 - width_err / self._cfg.width_tol_px)
            angle_diff = abs(angle_normalize(np.array([left_angle - right_angle]))[0])
            angle_score = max(0.0, 1.0 - angle_diff / self._cfg.angle_tol_rad)
            conf *= width_score * angle_score

        center_in_bounds = 0.0 <= center_x <= (self._cfg.image_width_px - 1.0)
        if not center_in_bounds:
            conf *= 0.5

        conf = float(np.clip(conf, 0.0, 1.0))
        return float(d_m), float(psi), conf

    def _line_x_at_y(self, line: Optional[Tuple[float, float, float, float]], y: float) -> Optional[float]:
        if line is None or any(v is None for v in line):
            return None
        vx, vy, x0, y0 = line
        if abs(vy) < 1e-6:
            return None
        return float(x0 + (vx / vy) * (y - y0))

    def _line_angle(self, line: Optional[Tuple[float, float, float, float]]) -> Optional[float]:
        if line is None or any(v is None for v in line):
            return None
        vx, vy = float(line[0]), float(line[1])
        if abs(vx) < 1e-8 and abs(vy) < 1e-8:
            return None
        if vy < 0:
            vx, vy = -vx, -vy
        return float(np.arctan2(vy, vx))

    def _lookahead_px(self) -> float:
        speed = self._target_speed_kph if self._target_speed_kph is not None else 0.0
        y = self._cfg.lookahead_px_base + self._cfg.lookahead_px_speed_gain * speed
        return float(np.clip(y, self._cfg.lookahead_px_min, self._cfg.lookahead_px_max))

    def _target_speed_to_mps(self) -> float:
        if self._target_speed_kph is None:
            return 0.0
        return float(self._target_speed_kph) * (1000.0 / 3600.0)

    def _update_output(self, t: float) -> None:
        if not self._initialized:
            return
        if self._last_output_time is None:
            self._x_out = self._x.copy()
            self._last_output_time = t
            self._lane_params_out = self._lane_from_state(self._x_out)
            return
        dt = t - self._last_output_time
        if dt <= 0.0:
            return
        alpha = 1.0
        if self._cfg.enable_output_smoothing:
            alpha = dt / (self._cfg.output_tau + dt)
        if not self._vision_ok:
            alpha *= 0.5
        self._x_out[0] += alpha * (self._x[0] - self._x_out[0])
        psi_err = angle_normalize(np.array([self._x[1] - self._x_out[1]]))[0]
        self._x_out[1] = angle_normalize(np.array([self._x_out[1] + alpha * psi_err]))[0]
        self._x_out[2] += alpha * (self._x[2] - self._x_out[2])
        self._last_output_time = t
        self._lane_params_out = self._lane_from_state(self._x_out)

    def _lane_from_state(self, x_state: np.ndarray) -> LaneParams:
        d_m, psi, _v = x_state
        pixel_to_meter = self._cfg.lane_width_m / self._cfg.lane_width_px
        d_px = d_m / pixel_to_meter
        y_ref = self._lookahead_px()
        center_x = self._cfg.lane_center_px + d_px
        center_x = float(np.clip(center_x, 0.0, self._cfg.image_width_px - 1.0))
        y_ref = float(np.clip(y_ref, 0.0, self._cfg.image_height_px - 1.0))

        angle = psi + np.pi / 2.0
        vx = float(np.cos(angle))
        vy = float(np.sin(angle))
        if vy < 0.0:
            vx, vy = -vx, -vy

        perp = np.array([-vy, vx], dtype=float)
        norm = np.hypot(perp[0], perp[1])
        if norm < 1e-6:
            perp = np.array([-1.0, 0.0])
            norm = 1.0
        perp /= norm

        half_w = self._cfg.lane_width_px / 2.0
        center = np.array([center_x, y_ref], dtype=float)
        left_point = center + perp * half_w
        right_point = center - perp * half_w

        left = (vx, vy, float(left_point[0]), float(left_point[1]))
        right = (vx, vy, float(right_point[0]), float(right_point[1]))
        return {"left": left, "right": right}

    def _geo_to_local(self, lat: float, lon: float) -> Tuple[float, float]:
        if self._gnss_origin is None:
            return 0.0, 0.0
        lat0, lon0 = self._gnss_origin
        r_earth = 6378137.0
        d_lat = np.deg2rad(lat - lat0)
        d_lon = np.deg2rad(lon - lon0)
        x = d_lon * np.cos(np.deg2rad(lat0)) * r_earth
        y = d_lat * r_earth
        return float(x), float(y)
