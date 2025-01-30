# pylint: skip-file
# flake8: noqa
# type: ignore
from dataclasses import dataclass


@dataclass
class WorkEnvConfig:
    seed: int = 42
    picture_path: str = "data/demo/earth_pro_poznan.jpg"
    patch_size: int = 51
    start: int = 0

@dataclass
class ParticlesConfig:
    number: int = 500
    rand_static_move: int = 20

@dataclass
class ResamplerConfig:
    name: str = "Systematic"

@dataclass
class MatcherConfig:
    name: str = "LBP"

@dataclass
class UAVConfig:
    traj_len: int = 300

@dataclass
class Config:
    work_env: WorkEnvConfig = WorkEnvConfig()
    particles: ParticlesConfig = ParticlesConfig()
    resampler: ResamplerConfig = ResamplerConfig()
    matcher: MatcherConfig = MatcherConfig()
    uav: UAVConfig = UAVConfig()
