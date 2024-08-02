from pathlib import Path

from . import constants

# Initialize all declared directories
for attribute in dir(constants):
    field_value = getattr(constants, attribute)
    if isinstance(field_value, Path) and field_value.stem == field_value.name:
        field_value.mkdir(exist_ok=True, parents=True)
