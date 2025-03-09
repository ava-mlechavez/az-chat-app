from typing import TypedDict, Annotated, Optional
from semantic_kernel.functions import kernel_function

class LightModel(TypedDict):
    id: int
    name: str
    is_on: bool | None
    brightness: int | None
    hex: str | None

class LightsPlugin:
    lights: list[LightModel] = [
        {"id": 1, "name": "Table Lamp", "is_on": False, "brightness": 100, "hex": "FF0000"},
      {"id": 2, "name": "Porch light", "is_on": False, "brightness": 50, "hex": "00FF00"},
      {"id": 3, "name": "Chandelier", "is_on": True, "brightness": 75, "hex": "0000FF"},
    ]

    @kernel_function(
            name="get_lights",
            description="Gets a list of lights and their current state.",
    )
    async def get_lights(self) -> list[LightModel]:
        """Gets a list of lights and their current state."""
        return self.lights

    @kernel_function(
            name="get_state",
            description="Gets the current state of a particular light.",
    )
    async def get_state(
            self,
            id: Annotated[int, "THe ID of the light"]
    ) -> Optional[LightModel]:
        """Gets the current state of a particular light."""
        filtered = filter(lambda x: x["id"] == id, self.lights)
        return filtered[0] if filtered else None


    @kernel_function(name="change_state", description="Changes the state of a particular light.")
    async def change_state(
        self,
        id: Annotated[int, "The ID of the light"],
        new_state: LightModel
    ) -> Optional[LightModel]:
        """Changes the state of the light."""
        for light in self.lights:
            if light["id"] == id:
                light["is_on"] = new_state.get("is_on", light["is_on"])
                light["brightness"] = new_state.get("brightness", light["brightness"])
                light["hex"] = new_state.get("hex", light["hex"])
                return light
        return None
