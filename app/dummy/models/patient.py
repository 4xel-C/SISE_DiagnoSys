from typing import Optional
from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class Patient:
    """Dummy patient dataclass"""

    id: int
    first_name: str
    last_name: str
    initials: Optional[str] = Field(default=None)

    def __post_init__(self):
        if not self.initials:
            # Auto compute initials
            self.initials = f"{self.first_name[0].upper()}{self.last_name[0].lower()}"
