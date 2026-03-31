#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastSLAM 1.0 — ROS 2 node
==========================
Implements the FastSLAM 1.0 algorithm (Montemerlo et al., 2002) for
landmark-based simultaneous localisation and mapping, ported 1-to-1 from
the ex07 assignment structure (ex7.py).

Algorithm per odom+scan cycle:
  1. Prediction  — sample new pose from the odometry motion model (r1, t, r2).
  2. Extraction  — detect point landmarks in /scan via spike detection.
  3. Association — nearest-neighbour data association (per particle).
  4. Correction  — EKF update on matched landmarks; initialise new ones.
  5. Weighting   — particle weight = product of per-landmark likelihoods.
  6. Resampling  — selective low-variance resampling (N_eff criterion).

Subscriptions:
  /scan    sensor_msgs/LaserScan   — laser range measurements  (articubot)
  /odom    nav_msgs/Odometry       — wheel odometry            (articubot)

Publications:
  /fast_slam/particles    geometry_msgs/PoseArray      — full particle cloud
  /fast_slam/pose         geometry_msgs/PoseStamped    — weighted-mean pose
  /fast_slam/landmarks    visualization_msgs/MarkerArray — landmark map (best particle)
  /fast_slam/trajectory   nav_msgs/Path                — best-particle trajectory

Motion model (Thrun et al., Table 5.6 — alpha-based):
  s_r1 = sqrt(alpha1*r1² + alpha2*t²)
  s_t  = sqrt(alpha3*t²  + alpha4*(r1²+r2²))
  s_r2 = sqrt(alpha1*r2² + alpha2*t²)

Landmark observation model (range-bearing):
  h = [sqrt((lx-rx)²+(ly-ry)²),  atan2(ly-ry, lx-rx) - theta]
  H = Jacobian of h w.r.t. landmark position

EKF initialisation (new landmark):
  mu    = inverse_sensor_model(pose, z_range, z_bearing)
  sigma = H⁻¹ · Q · H⁻ᵀ       (Q = sensor noise covariance)

EKF update (known landmark):
  S = H · sigma · Hᵀ + Q
  K = sigma · Hᵀ · S⁻¹
  mu    ← mu + K · (z − ẑ)
  sigma ← (I − K·H) · sigma
  w_i  *= N(z; ẑ, S)            (Gaussian likelihood)
"""

import copy
import os
from math import atan2, cos, sin, sqrt, pi

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import (
    PoseArray, Pose, PoseStamped, Point, Quaternion, TransformStamped,
)
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
import tf_transformations


# ──────────────────────────────────────────────────────────────────────────────
#  Maths helpers
# ──────────────────────────────────────────────────────────────────────────────

def wrap_to_pi(theta: float) -> float:
    return (theta + pi) % (2.0 * pi) - pi


def yaw_to_quat(yaw: float) -> Quaternion:
    q = tf_transformations.quaternion_from_euler(0.0, 0.0, float(yaw))
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


def quat_to_yaw(q) -> float:
    return tf_transformations.euler_from_quaternion(
        [q.x, q.y, q.z, q.w])[2]


# ──────────────────────────────────────────────────────────────────────────────
#  Per-landmark EKF  (mirrors Particle.LandmarkEKF from ex7.py)
# ──────────────────────────────────────────────────────────────────────────────

class LandmarkEKF:
    """2-D landmark estimate: position mean + 2×2 covariance."""

    __slots__ = ('observed', 'mu', 'sigma')

    def __init__(self):
        self.observed = False
        self.mu       = np.zeros(2)            # [x, y] world frame
        self.sigma    = np.eye(2) * 1e4        # large initial uncertainty


# ──────────────────────────────────────────────────────────────────────────────
#  Particle  (mirrors Particle from ex7.py, adapted for dynamic landmarks)
# ──────────────────────────────────────────────────────────────────────────────

class Particle:
    """
    FastSLAM particle — robot pose + per-landmark EKF map + weight.

    Differences from ex7.py:
    - landmarks stored in a dict (dynamic, unbounded up to max_landmarks)
      rather than a fixed-size list, because landmark IDs are discovered
      online via data association rather than given by the dataset.
    - Data association is performed inside correction_step (per particle),
      using Euclidean NN in world frame with a configurable threshold.
    - Motion noise uses the Thrun alpha model (scales with motion magnitude).
    """

    def __init__(
        self,
        init_pose: np.ndarray,        # [x, y, theta]
        alpha: list,                  # [a1, a2, a3, a4]  motion noise alphas
        Q_diag: np.ndarray,           # [sigma_r², sigma_b²] sensor noise variances
        assoc_dist: float,            # NN association threshold [m]
        max_landmarks: int,
    ):
        self.pose        = np.array(init_pose, dtype=float)  # [x, y, theta]
        self.weight      = 1.0
        self.trajectory  = []                 # list of pose snapshots

        self._alpha       = alpha
        self._Q           = np.diag(Q_diag)  # 2×2 sensor noise covariance
        self._assoc_dist  = assoc_dist
        self._max_lm      = max_landmarks
        self._traj_max_len = 500

        self.landmarks   = {}                 # lm_id (int) → LandmarkEKF
        self._next_id    = 0

    # ── prediction step (Q1 in ex7.py) ───────────────────────────────────────

    def prediction_step(self, r1: float, t: float, r2: float) -> None:
        """Sample new pose from noisy odometry motion model (Thrun Table 5.6)."""
        # Cap stored trajectory to avoid unbounded deepcopy cost during resampling
        if len(self.trajectory) >= self._traj_max_len:
            self.trajectory = self.trajectory[-(self._traj_max_len // 2):]
        self.trajectory.append(self.pose.copy())

        a1, a2, a3, a4 = self._alpha

        s_r1 = sqrt(max(a1 * r1**2 + a2 * t**2,  1e-12))
        s_t  = sqrt(max(a3 * t**2  + a4 * (r1**2 + r2**2), 1e-12))
        s_r2 = sqrt(max(a1 * r2**2 + a2 * t**2,  1e-12))

        r1_hat = r1 - np.random.normal(0.0, s_r1)
        t_hat  = t  - np.random.normal(0.0, s_t)
        r2_hat = r2 - np.random.normal(0.0, s_r2)

        x, y, th = self.pose
        self.pose = np.array([
            x  + t_hat * cos(th + r1_hat),
            y  + t_hat * sin(th + r1_hat),
            wrap_to_pi(th + r1_hat + r2_hat),
        ])

    # ── measurement model (Q2 in ex7.py) ─────────────────────────────────────

    def _measurement_model(self, lm: LandmarkEKF):
        """
        Expected measurement h and Jacobian H for one landmark.

        Returns:
          h : (2,)    [expected_range, expected_bearing]
          H : (2, 2)  d(h)/d(landmark_position)
        """
        x, y, th = self.pose
        dx = lm.mu[0] - x
        dy = lm.mu[1] - y
        q  = max(dx**2 + dy**2, 1e-9)
        r_hat = sqrt(q)
        b_hat = wrap_to_pi(atan2(dy, dx) - th)

        h = np.array([r_hat, b_hat])
        H = np.array([
            [ dx / r_hat,  dy / r_hat],
            [-dy / q,      dx / q   ],
        ])
        return h, H

    # ── inverse sensor model (used for landmark initialisation) ──────────────

    def _inv_sensor_model(self, z_range: float, z_bearing: float) -> np.ndarray:
        """Convert range-bearing obs to world-frame position estimate."""
        x, y, th = self.pose
        return np.array([
            x + z_range * cos(th + z_bearing),
            y + z_range * sin(th + z_bearing),
        ])

    # ── correction step (Q3 in ex7.py) ───────────────────────────────────────

    def correction_step(self, raw_obs: list) -> None:
        """
        Update landmark EKFs and accumulate weight for this particle.

        raw_obs: list of (z_range, z_bearing) in robot frame — these are the
                 extracted landmark observations from the laser scan.
                 Data association is performed per-particle (NN in world frame).
        """
        joint_likelihood = 1.0
        x, y, th = self.pose

        for z_r, z_b in raw_obs:
            # --- Data association (per particle) ---
            # Convert observation to world frame for NN search.
            obs_wx = x + z_r * cos(th + z_b)
            obs_wy = y + z_r * sin(th + z_b)

            best_id   = None
            best_dist = float('inf')
            for lm_id, lm in self.landmarks.items():
                if not lm.observed:
                    continue
                d = sqrt((obs_wx - lm.mu[0])**2 + (obs_wy - lm.mu[1])**2)
                if d < best_dist:
                    best_dist = d
                    best_id   = lm_id

            if best_dist <= self._assoc_dist:
                lm_id = best_id
            else:
                if len(self.landmarks) >= self._max_lm:
                    continue  # map is full — skip
                lm_id = self._next_id
                self._next_id += 1
                self.landmarks[lm_id] = LandmarkEKF()

            lm = self.landmarks[lm_id]

            # --- Initialise new landmark (Q3 A in ex7.py) ---
            if not lm.observed:
                lm.mu = self._inv_sensor_model(z_r, z_b)
                _, H  = self._measurement_model(lm)   # Jacobian at init pos
                H_inv = np.linalg.inv(H)
                lm.sigma   = H_inv @ self._Q @ H_inv.T
                lm.observed = True
                # Weight unchanged for newly initialised landmarks

            # --- EKF correction on known landmark (Q3 B / Q3 C in ex7.py) ---
            else:
                h_hat, H = self._measurement_model(lm)

                # Innovation covariance  S = H·Σ·Hᵀ + Q
                S = H @ lm.sigma @ H.T + self._Q

                # Kalman gain  K = Σ·Hᵀ·S⁻¹
                S_inv = np.linalg.inv(S)
                K     = lm.sigma @ H.T @ S_inv

                # Innovation  δz = z − ẑ  (bearing wrapped to [−π, π])
                dz    = np.array([z_r - h_hat[0],
                                  wrap_to_pi(z_b - h_hat[1])])

                # Update mean and covariance
                lm.mu    = lm.mu + K @ dz
                lm.sigma = (np.eye(2) - K @ H) @ lm.sigma

                # Gaussian likelihood  p(z | μ, S)
                det_S = float(np.linalg.det(S))
                if det_S > 1e-300:
                    exponent    = float(-0.5 * dz @ S_inv @ dz)
                    likelihood  = (1.0 / (2.0 * pi * sqrt(abs(det_S)))) * np.exp(exponent)
                    joint_likelihood *= max(likelihood, 1e-300)

        self.weight *= joint_likelihood


# ──────────────────────────────────────────────────────────────────────────────
#  Low-variance resampler  (Probabilistic Robotics, p. 109 / ex7.py)
# ──────────────────────────────────────────────────────────────────────────────

def low_variance_resample(particles: list) -> list:
    """
    Low-variance (systematic) resampling.
    Returns a new list of deep-copied particles with uniform weights.
    """
    N = len(particles)
    weights = np.array([p.weight for p in particles], dtype=float)
    w_sum = weights.sum()
    if w_sum < 1e-300:
        weights = np.ones(N) / N
    else:
        weights /= w_sum

    r      = np.random.uniform(0.0, 1.0 / N)
    cumsum = np.cumsum(weights)
    i      = 0
    new_particles = []

    for m in range(N):
        U = r + m / N
        while U > cumsum[i] and i < N - 1:
            i += 1
        p_copy = copy.deepcopy(particles[i])
        p_copy.weight = 1.0 / N
        new_particles.append(p_copy)

    return new_particles


# ──────────────────────────────────────────────────────────────────────────────
#  Landmark extraction from LaserScan
# ──────────────────────────────────────────────────────────────────────────────

def extract_landmarks(
    ranges: np.ndarray,
    angles: np.ndarray,
    max_range: float = 10.0,
    min_range: float = 0.15,
    spike_threshold: float = 0.3,
    gap_threshold: float = 0.5,
    max_obs: int = 20,
) -> list:
    """
    Detect point-landmark candidates from a laser scan using two detectors:

    1. Spike / pillar detector  — beam[i] is much shorter than both neighbours.
       Finds thin poles, chair legs, and narrow objects.

    2. Range-gap endpoint detector — large depth jump between consecutive beams.
       The nearer endpoint of each gap is a good corner/edge landmark.
       This is far more common than spikes in typical indoor environments.

    Returns up to max_obs (z_range, z_bearing) tuples in the robot frame,
    evenly sampled if more candidates are found than the cap.
    """
    n     = len(ranges)
    valid = np.isfinite(ranges) & (ranges > min_range) & (ranges < max_range)
    obs   = []

    for i in range(1, n - 1):
        if not valid[i]:
            continue
        r     = ranges[i]
        left  = ranges[i - 1] if valid[i - 1] else max_range
        right = ranges[i + 1] if valid[i + 1] else max_range

        # Spike: isolated protrusion shorter than both neighbours
        if r < (left - spike_threshold) and r < (right - spike_threshold):
            obs.append((float(r), float(angles[i])))
            continue

        # Gap endpoint: large depth discontinuity on the right side
        # → the closer beam is a corner / object edge
        if valid[i + 1]:
            gap = abs(ranges[i + 1] - r)
            if gap > gap_threshold:
                # Take the nearer of the two endpoints
                if r <= ranges[i + 1]:
                    obs.append((float(r), float(angles[i])))
                else:
                    obs.append((float(ranges[i + 1]), float(angles[i + 1])))

    # If too many candidates, sub-sample uniformly to cap
    if len(obs) > max_obs:
        step = len(obs) / max_obs
        obs  = [obs[int(k * step)] for k in range(max_obs)]

    return obs


# ──────────────────────────────────────────────────────────────────────────────
#  ROS 2 node
# ──────────────────────────────────────────────────────────────────────────────

class FastSLAMNode(Node):

    def __init__(self):
        super().__init__('fast_slam_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('num_particles',        50)
        # Motion model — Thrun alpha noise (Table 5.6)
        self.declare_parameter('alpha1',               0.1)   # rot noise ← rot
        self.declare_parameter('alpha2',               0.1)   # rot noise ← trans
        self.declare_parameter('alpha3',               0.1)   # trans noise ← trans
        self.declare_parameter('alpha4',               0.1)   # trans noise ← rot
        # Sensor noise (std devs → stored as variances in Q)
        self.declare_parameter('sensor_noise_range',   0.2)   # σ_r  [m]
        self.declare_parameter('sensor_noise_bearing', 0.1)   # σ_b  [rad]
        # Landmark extraction
        self.declare_parameter('spike_threshold',      0.3)   # spike depth gap [m]
        self.declare_parameter('gap_threshold',        0.5)   # range-jump depth gap [m]
        self.declare_parameter('max_obs_per_scan',    20)     # cap observations per scan
        self.declare_parameter('max_scan_range',      10.0)   # ignore beams > this [m]
        self.declare_parameter('min_scan_range',       0.15)  # ignore beams < this [m]
        # Data association
        self.declare_parameter('assoc_dist',           1.0)   # NN threshold [m]
        self.declare_parameter('max_landmarks',       100)    # per-particle cap
        # Resampling
        self.declare_parameter('neff_ratio',           0.5)   # resample when N_eff < ratio*N
        # Motion gate: skip update when robot has not moved enough
        self.declare_parameter('min_trans',            0.01)  # [m]
        self.declare_parameter('min_rot',              0.02)  # [rad]
        # Output
        self.declare_parameter('frame_id',           'odom')
        self.declare_parameter('traj_max_len',        500)    # trajectory history
        # Map saving
        self.declare_parameter('save_map_path',
                               '~/robotics/ros2_ws/maps/fast_slam_landmarks')
        self.declare_parameter('auto_save_interval',  0.0)    # 0 = shutdown only

        p = lambda name: self.get_parameter(name).value

        self._num_particles = p('num_particles')
        self._alpha         = [p('alpha1'), p('alpha2'), p('alpha3'), p('alpha4')]
        # Q diagonal: [sigma_r², sigma_b²]
        self._Q_diag        = np.array([p('sensor_noise_range')**2,
                                        p('sensor_noise_bearing')**2])
        self._spike_thr          = p('spike_threshold')
        self._gap_thr            = p('gap_threshold')
        self._max_obs_per_scan   = p('max_obs_per_scan')
        self._max_range          = p('max_scan_range')
        self._min_range          = p('min_scan_range')
        self._assoc_dist    = p('assoc_dist')
        self._max_lm        = p('max_landmarks')
        self._neff_ratio    = p('neff_ratio')
        self._min_trans     = p('min_trans')
        self._min_rot       = p('min_rot')
        self._frame_id      = p('frame_id')
        self._traj_max_len  = p('traj_max_len')

        # ── Internal state ────────────────────────────────────────────────────
        self._particles       = None   # list[Particle]
        self._prev_odom       = None   # [x, y, theta]  — latest odom
        self._ref_odom        = None   # [x, y, theta]  — pose at last update
        self._initialized     = False

        # Latest wheel velocities — read from twist in odom msg
        self._last_lin_vel    = 0.0
        self._last_ang_vel    = 0.0
        # Gate thresholds: skip filter update when robot is truly stationary
        self._LIN_VEL_THR     = 0.01   # m/s
        self._ANG_VEL_THR     = 0.01   # rad/s

        # Warmup: ignore first N odom messages while Gazebo physics settles
        self._warmup_count    = 0
        self._WARMUP_N        = 50

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Odometry,  '/odom', self._odom_cb,  10)
        self.create_subscription(LaserScan, '/scan', self._scan_cb,  10)

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_particles  = self.create_publisher(PoseArray,    '/fast_slam/particles',  10)
        self._pub_pose       = self.create_publisher(PoseStamped,  '/fast_slam/pose',       10)
        self._pub_landmarks  = self.create_publisher(MarkerArray,  '/fast_slam/landmarks',  10)
        self._pub_trajectory = self.create_publisher(Path,         '/fast_slam/trajectory', 10)

        # ── TF broadcaster (map → odom) ───────────────────────────────────────
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ── Optional periodic landmark-map save ───────────────────────────────
        auto_save = self.get_parameter('auto_save_interval').value
        if auto_save > 0.0:
            self.create_timer(auto_save, self.save_map)

        self.get_logger().info(
            f'FastSLAM node started — {self._num_particles} particles, '
            f'assoc_dist={self._assoc_dist} m, '
            f'spike_thr={self._spike_thr} m'
        )

    # ── Odometry callback ──────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        yaw = quat_to_yaw(msg.pose.pose.orientation)
        pose = np.array([pos.x, pos.y, yaw])

        # Store current velocity — used as the primary motion gate in _scan_cb
        self._last_lin_vel = float(np.hypot(msg.twist.twist.linear.x,
                                            msg.twist.twist.linear.y))
        self._last_ang_vel = float(abs(msg.twist.twist.angular.z))

        # Warmup: absorb unstable initial Gazebo messages, then seed state
        if self._warmup_count < self._WARMUP_N:
            self._warmup_count += 1
            self._prev_odom = pose
            return

        self._prev_odom = pose

        # Initialise particles once on the first settled odom
        if not self._initialized:
            self._init_particles(pose)
            self._ref_odom   = pose.copy()
            self._initialized = True

    # ── Scan callback — main FastSLAM update ──────────────────────────────────

    def _scan_cb(self, msg: LaserScan) -> None:
        if not self._initialized or self._particles is None:
            return
        if self._prev_odom is None or self._ref_odom is None:
            return

        stamp = msg.header.stamp

        # ── Gate 1: velocity — skip when robot is genuinely stationary ────────
        # Reading twist directly is the most reliable signal; accumulated pose
        # diffs accumulate floating-point noise even when wheels aren't turning.
        if (self._last_lin_vel < self._LIN_VEL_THR and
                self._last_ang_vel < self._ANG_VEL_THR):
            self._publish(stamp)
            return

        # ── Odometry delta → (r1, t, r2) ─────────────────────────────────────
        cur  = self._prev_odom
        prev = self._ref_odom
        dx   = cur[0] - prev[0]
        dy   = cur[1] - prev[1]
        t    = float(np.hypot(dx, dy))

        # Guard atan2 against zero translation: when t ≈ 0, atan2(0, 0)
        # returns 0 but (0 - prev_theta) can be a large spurious rotation that
        # bypasses the gate and teleports every particle.
        if t > 1e-6:
            r1 = float(wrap_to_pi(np.arctan2(dy, dx) - prev[2]))
        else:
            r1 = 0.0
        r2 = float(wrap_to_pi(cur[2] - prev[2] - r1))

        # ── Gate 2: pose-diff — skip accumulated delta below sensor noise ─────
        if t < self._min_trans and abs(r1) + abs(r2) < self._min_rot:
            self._publish(stamp)
            return

        self._ref_odom = cur.copy()

        # ── Landmark extraction ───────────────────────────────────────────────
        n_beams = len(msg.ranges)
        angles  = np.linspace(msg.angle_min, msg.angle_max, n_beams)
        ranges  = np.array(msg.ranges, dtype=np.float64)
        raw_obs = extract_landmarks(
            ranges, angles,
            max_range=self._max_range,
            min_range=self._min_range,
            spike_threshold=self._spike_thr,
            gap_threshold=self._gap_thr,
            max_obs=self._max_obs_per_scan,
        )

        # ── FastSLAM prediction step ──────────────────────────────────────────
        for p in self._particles:
            p.prediction_step(r1, t, r2)

        # ── FastSLAM correction step (per particle, includes per-particle DA) ─
        if raw_obs:
            for p in self._particles:
                p.correction_step(raw_obs)

        # ── Normalise weights ─────────────────────────────────────────────────
        weights = np.array([p.weight for p in self._particles], dtype=float)
        w_sum   = weights.sum()
        if w_sum < 1e-300:
            weights = np.ones(self._num_particles) / self._num_particles
        else:
            weights /= w_sum
        for i, p in enumerate(self._particles):
            p.weight = float(weights[i])

        # ── Selective resampling (N_eff criterion — Q4 in ex7.py) ────────────
        N_eff = 1.0 / float(np.sum(weights ** 2))
        if N_eff < self._neff_ratio * self._num_particles:
            self._particles = low_variance_resample(self._particles)
            self.get_logger().debug(
                f'Resampled — N_eff={N_eff:.1f} < '
                f'{self._neff_ratio * self._num_particles:.0f}'
            )

        self._publish(stamp)

    # ── Particle initialisation ───────────────────────────────────────────────

    def _init_particles(self, init_pose: np.ndarray) -> None:
        """Scatter particles in a tight Gaussian around the initial odom pose."""
        self._particles = []
        for _ in range(self._num_particles):
            pose = init_pose.copy()
            pose[0] += np.random.normal(0.0, 0.05)
            pose[1] += np.random.normal(0.0, 0.05)
            pose[2]  = wrap_to_pi(pose[2] + np.random.normal(0.0, 0.02))
            self._particles.append(
                Particle(
                    pose,
                    self._alpha,
                    self._Q_diag,
                    self._assoc_dist,
                    self._max_lm,
                )
            )
        self.get_logger().info(
            f'Initialised {self._num_particles} particles at '
            f'({init_pose[0]:.2f}, {init_pose[1]:.2f}, '
            f'{np.degrees(init_pose[2]):.1f}°)'
        )

    # ── Weighted mean pose ────────────────────────────────────────────────────

    def _mean_pose(self):
        """Compute importance-weighted mean pose (x, y, theta)."""
        weights = np.array([p.weight for p in self._particles], dtype=float)
        w_sum   = weights.sum()
        if w_sum < 1e-300:
            weights = np.ones(self._num_particles) / self._num_particles
            w_sum   = 1.0
        weights /= w_sum

        mean_x  = float(np.dot(weights, [p.pose[0] for p in self._particles]))
        mean_y  = float(np.dot(weights, [p.pose[1] for p in self._particles]))
        # Circular mean for angle
        c_sum = float(np.dot(weights, [cos(p.pose[2]) for p in self._particles]))
        s_sum = float(np.dot(weights, [sin(p.pose[2]) for p in self._particles]))
        mean_th = atan2(s_sum, c_sum)
        return mean_x, mean_y, mean_th

    def _best_particle(self) -> Particle:
        return max(self._particles, key=lambda p: p.weight)

    # ── Publishers ────────────────────────────────────────────────────────────

    def _publish(self, stamp) -> None:
        self._publish_particles(stamp)
        self._publish_pose(stamp)
        self._publish_landmarks(stamp)
        self._publish_trajectory(stamp)
        self._broadcast_tf(stamp)

    def _publish_particles(self, stamp) -> None:
        msg = PoseArray()
        msg.header = Header(frame_id=self._frame_id, stamp=stamp)
        for p in self._particles:
            pose = Pose()
            pose.position.x = float(p.pose[0])
            pose.position.y = float(p.pose[1])
            pose.position.z = 0.0
            pose.orientation = yaw_to_quat(p.pose[2])
            msg.poses.append(pose)
        self._pub_particles.publish(msg)

    def _publish_pose(self, stamp) -> None:
        x, y, th = self._mean_pose()
        msg = PoseStamped()
        msg.header = Header(frame_id=self._frame_id, stamp=stamp)
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        msg.pose.orientation = yaw_to_quat(th)
        self._pub_pose.publish(msg)

    def _publish_landmarks(self, stamp) -> None:
        """
        Visualise landmark estimates from the highest-weight particle.
        Each landmark: red sphere (position) + text (ID).
        """
        best = self._best_particle()
        markers = MarkerArray()

        # Clear previous markers
        clear = Marker()
        clear.header = Header(frame_id=self._frame_id, stamp=stamp)
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        for lm_id, lm in best.landmarks.items():
            if not lm.observed:
                continue

            # Sphere at landmark mean position
            sphere = Marker()
            sphere.header    = Header(frame_id=self._frame_id, stamp=stamp)
            sphere.ns        = 'lm_sphere'
            sphere.id        = lm_id
            sphere.type      = Marker.SPHERE
            sphere.action    = Marker.ADD
            sphere.pose.position.x = float(lm.mu[0])
            sphere.pose.position.y = float(lm.mu[1])
            sphere.pose.position.z = 0.15
            sphere.pose.orientation.w = 1.0
            sphere.scale.x   = 0.25
            sphere.scale.y   = 0.25
            sphere.scale.z   = 0.25
            sphere.color.r   = 1.0
            sphere.color.g   = 0.2
            sphere.color.b   = 0.2
            sphere.color.a   = 0.85
            markers.markers.append(sphere)

            # Text label
            text = Marker()
            text.header      = Header(frame_id=self._frame_id, stamp=stamp)
            text.ns          = 'lm_text'
            text.id          = lm_id
            text.type        = Marker.TEXT_VIEW_FACING
            text.action      = Marker.ADD
            text.pose.position.x = float(lm.mu[0])
            text.pose.position.y = float(lm.mu[1])
            text.pose.position.z = 0.45
            text.pose.orientation.w = 1.0
            text.scale.z     = 0.20
            text.color.r     = 1.0
            text.color.g     = 1.0
            text.color.b     = 1.0
            text.color.a     = 1.0
            text.text        = str(lm_id)
            markers.markers.append(text)

        self._pub_landmarks.publish(markers)

    def _publish_trajectory(self, stamp) -> None:
        """Publish the trajectory of the highest-weight particle."""
        best = self._best_particle()
        msg  = Path()
        msg.header = Header(frame_id=self._frame_id, stamp=stamp)

        history = best.trajectory[-self._traj_max_len:]
        for pose_arr in history:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(pose_arr[0])
            ps.pose.position.y = float(pose_arr[1])
            ps.pose.position.z = 0.0
            ps.pose.orientation = yaw_to_quat(pose_arr[2])
            msg.poses.append(ps)

        self._pub_trajectory.publish(msg)

    def _broadcast_tf(self, stamp) -> None:
        """
        Broadcast map → odom transform so the estimated pose appears in RViz2.

        With FastSLAM building its map in the odom frame from scratch, the
        map→odom correction starts at identity and accumulates as the SLAM
        estimate diverges from raw odometry.  The correction is:

            T_map_odom = T_map_robot × inv(T_odom_robot)

        where T_map_robot  = weighted-mean pose from FastSLAM
              T_odom_robot = latest raw odometry pose
        """
        if self._prev_odom is None:
            return

        mx, my, mth = self._mean_pose()
        ox, oy, oth  = self._prev_odom

        cm, sm = cos(mth), sin(mth)
        T_map_robot  = np.array([[cm, -sm, mx],
                                  [sm,  cm, my],
                                  [0,   0,  1]])

        co, so = cos(oth), sin(oth)
        T_odom_robot = np.array([[co, -so, ox],
                                  [so,  co, oy],
                                  [0,   0,  1]])

        T_map_odom = T_map_robot @ np.linalg.inv(T_odom_robot)

        tx  = T_map_odom[0, 2]
        ty  = T_map_odom[1, 2]
        tth = atan2(T_map_odom[1, 0], T_map_odom[0, 0])

        t = TransformStamped()
        t.header.stamp    = stamp
        t.header.frame_id = 'map'
        t.child_frame_id  = self._frame_id    # e.g. 'odom'
        t.transform.translation.x = tx
        t.transform.translation.y = ty
        t.transform.translation.z = 0.0
        q = tf_transformations.quaternion_from_euler(0.0, 0.0, tth)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self._tf_broadcaster.sendTransform(t)


    # ── Map save ──────────────────────────────────────────────────────────────

    def save_map(self) -> None:
        """
        Save the landmark map from the highest-weight particle to disk as a
        YAML file.  Each entry contains the landmark ID, 2-D world-frame
        position (mu) and 2×2 covariance matrix (sigma).

        The file is written to save_map_path + '.yaml' and is compatible with
        the world.data format used in the ex07 assignment for inspection, as
        well as being loadable by custom downstream code.
        """
        import yaml

        if self._particles is None:
            self.get_logger().warn('save_map called but particles not initialised yet — skipping.')
            return

        path = os.path.expanduser(
            self.get_parameter('save_map_path').value) + '.yaml'
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        best = self._best_particle()
        mx, my, mth = self._mean_pose()

        landmarks_out = []
        for lm_id in sorted(best.landmarks.keys()):
            lm = best.landmarks[lm_id]
            if not lm.observed:
                continue
            landmarks_out.append({
                'id':    int(lm_id),
                'mu':    [round(float(lm.mu[0]), 4),
                          round(float(lm.mu[1]), 4)],
                'sigma': [[round(float(lm.sigma[0, 0]), 6),
                           round(float(lm.sigma[0, 1]), 6)],
                          [round(float(lm.sigma[1, 0]), 6),
                           round(float(lm.sigma[1, 1]), 6)]],
            })

        data = {
            'frame_id':      self._frame_id,
            'num_landmarks': len(landmarks_out),
            'best_pose':     {
                'x': round(mx, 4),
                'y': round(my, 4),
                'theta_deg': round(float(np.degrees(mth)), 2),
            },
            'landmarks': landmarks_out,
        }

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        self.get_logger().info(
            f'Landmark map saved → {path}  '
            f'({len(landmarks_out)} landmarks)'
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = FastSLAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_map()        # always flush landmark map on exit
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
