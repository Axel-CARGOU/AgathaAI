from __future__ import annotations
import logging

from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("app")

@dataclass(frozen=True)
class VTSParameter:
    name: str
    explanation: str
    min_value: float
    max_value: float
    default_value: float
    smoothing: float


CUSTOM_PARAMETERS: tuple[VTSParameter, ...] = (
    VTSParameter("ToggleScaryFace", "Scary face toggle", 0.0, 1.0, 0.0, 0.0),
    VTSParameter("MoveAngleX", "Angle X Movements", -30.0, 30.0, 0.0, 1.0),
    VTSParameter("MoveAngleY", "Angle Y Movements", -30.0, 30.0, 0.0, 1.0),
    VTSParameter("MoveAngleZ", "Angle Z Movements", -30.0, 30.0, 0.0, 1.0),
    VTSParameter("MoveBrowLY", "Left Brow Y Movements", -1.0, 1.0, 0.0, 1.0),
    VTSParameter("MoveBrowRY", "Right Brow Y Movements", -1.0, 1.0, 0.0, 1.0),
    VTSParameter("FormBrowLY", "Left Brow Y Form", -1.0, 1.0, 0.0, 1.0),
    VTSParameter("FormBrowRY", "Right Brow Y Form", -1.0, 1.0, 0.0, 1.0),
    VTSParameter("MoveEyeBallsX", "Eyeballs X Movements", -1.0, 1.0, 0.0, 1.0),
    VTSParameter("MoveEyeBallsY", "Eyeballs Y Movements", -1.0, 1.0, 0.0, 1.0),
    VTSParameter("OpenEyes", "Eyes Open/Close", 0.0, 1.0, 1.0, 1.0),
    VTSParameter("OpenMouth", "Mouth Open", 0.0, 1.0, 0.0, 1.0),
    VTSParameter("FormMouth", "Mouth Form", -1.0, 1.0, 0.0, 1.0),
    VTSParameter("MoveBodyX", "Body X Movements", -10.0, 10.0, 0.0, 1.0),
)


def make_parameter_creation_request(param: VTSParameter) -> dict[str, Any]:
    return {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": f"create_custom_param_{param.name}",
        "messageType": "ParameterCreationRequest",
        "data": {
            "parameterName": param.name,
            "explanation": param.explanation,
            "min": param.min_value,
            "max": param.max_value,
            "defaultValue": param.default_value,
            "smoothing": param.smoothing,
            "deleteIfPluginIsUnloaded": True,
        },
    }


async def create_all_tracking_parameters(vts) -> None:
    for param in CUSTOM_PARAMETERS:
        logger.info(f"[VTUBER] Creating custom tracking parameter: {param.name}")
        await vts.request(make_parameter_creation_request(param))

    logger.info("\n\n[VTUBER] Custom tracking parameters successfully created.\n")
