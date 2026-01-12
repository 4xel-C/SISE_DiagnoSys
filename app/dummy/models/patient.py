from typing import Optional
from pydantic import Field
from pydantic.dataclasses import dataclass

import random





@dataclass
class Patient:
    """Dummy patient dataclass"""
    
    id: int
    first_name: str
    last_name: str
    initials: Optional[str] = Field(default=None)
    color: Optional[str] = Field(default=None)

    def __post_init__(self):
        if not self.initials:
            # Auto compute initials
            self.initials = f"{self.first_name[0].upper()}{self.last_name[0].upper()}"
        if not self.color:
            # Generate random color
            self.color = f"hsl({random.randint(0, 360)} 92% 80%)"