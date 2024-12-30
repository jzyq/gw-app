from pydantic import BaseModel, Field
from typing import List


class Vector2(BaseModel):
    x: int
    y: int


class Position(BaseModel):
    areas: List[Vector2]


class InferenceObject(BaseModel):
    object_id: str = Field(alias="objectId")
    type_list: List[str] = Field(alias="typeList")
    image_url_list: List[str] = Field(alias="imageUrlList")
    image_normal_url_path: str = Field(alias="imageNormalUrlPath")
    pos: List[Position]


class CreateInferenceTaskRequest(BaseModel):
    request_host_ip: str = Field(alias="requestHostIp")
    request_host_port: str = Field(alias="requestHostPort")
    request_id: str = Field(alias="requestId")
    object_list: List[InferenceObject] = Field(alias="objectList")
