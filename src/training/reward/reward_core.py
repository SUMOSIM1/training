import importlib
from abc import ABC, abstractmethod
from enum import Enum


import training.simrunner_core as src


class RewardHandlerName(str, Enum):
    CONTINUOUS_CONSIDER_ALL = "continuous-consider-all"
    END_CONSIDER_ALL = "end-consider-all"
    REDUCED_PUSH_REWARD = "reduced-push-reward"
    SPEED_BONUS = "speed-bonus"
    CAN_SEE = "can-see"


class RewardHandler(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def calculate_reward(self, state: src.SimulationState) -> tuple[float, float]:
        pass

    @abstractmethod
    def calculate_end_reward(
        self,
        states: list[src.SimulationState],
        properties1: list[list],
        properties2: list[list],
        max_simulation_count: int,
    ) -> tuple[float, float]:
        pass


class RewardHandlerProvider:
    @staticmethod
    def get(name: RewardHandlerName) -> RewardHandler:
        match name:
            case RewardHandlerName.CONTINUOUS_CONSIDER_ALL:
                module = importlib.import_module("training.reward.reward")
                class_ = module.ConsiderAllRewardHandler
                return class_()
            case RewardHandlerName.REDUCED_PUSH_REWARD:
                module = importlib.import_module("training.reward.reward")
                class_ = module.ReducedPushRewardHandler
                return class_()
            case RewardHandlerName.SPEED_BONUS:
                module = importlib.import_module("training.reward.reward")
                class_ = module.SpeedBonusRewardHandler
                return class_()
            case RewardHandlerName.CAN_SEE:
                module = importlib.import_module("training.reward.reward")
                class_ = module.CanSeeRewardHandler
                return class_()
            case _:
                raise RuntimeError(f"Unknown reward handler {name}")
