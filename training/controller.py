from dataclasses import dataclass
from enum import Enum


class SectorName(Enum):
    UNDEF = "undef"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


@dataclass
class CombiSensor:
    left_distance: float
    front_distance: float
    right_distance: float
    opponent_in_sector: SectorName


@dataclass
class DiffDriveValues:
    right_velo: float
    left_velo: float


class Controller:
    def take_step(self, sensor: CombiSensor) -> DiffDriveValues:
        pass

    def name(self) -> str:
        pass

    def description(self) -> dict:
        pass


class ControllerProvider:
    @staticmethod
    def get(name: str) -> Controller:
        match name:
            case "fast-circle":
                return FastCircleController()
            case "slow-circle":
                return FastCircleController()
            case _:
                raise RuntimeError(f"Unknown controller {name}")


def _circle_description(left: float, right: float) -> dict:
    return {
        "description": "Drives with a constant speed for each wheel",
        "left wheel": left,
        "right wheel": right,
    }


class FastCircleController(Controller):
    left_wheel = 0.5
    right_wheel = 0.5

    def take_step(self, sensor: CombiSensor) -> DiffDriveValues:
        return DiffDriveValues(self.left_wheel, self.right_wheel)

    def name(self) -> str:
        return "fast-circle"

    def description(self) -> dict:
        return _circle_description(self.left_wheel, self.right_wheel)


class SlowCircleController(Controller):
    left_wheel = 0.3
    right_wheel = 0.2

    def take_step(self, sensor: CombiSensor) -> DiffDriveValues:
        return DiffDriveValues(self.left_wheel, self.right_wheel)

    def name(self) -> str:
        return "slow-circle"

    def description(self) -> dict:
        return _circle_description(self.left_wheel, self.right_wheel)
