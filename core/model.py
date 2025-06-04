from typing import Literal
from pydantic import BaseModel
from playwright.async_api import Page

class JSOutput(BaseModel):
    script: str
    change_description: str

class PlaywrightSession(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    page: Page

class Evaluation(BaseModel):
    feedback: str
    score: Literal["pass", "fail"]

class ImageChange(BaseModel):
    image_url: str
    change_description: str
