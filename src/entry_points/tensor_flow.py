from fastapi import APIRouter

from src.service_layer.tensor_flow import TensorFlow
from pydantic import BaseModel

router = APIRouter(
    prefix="/train",
    responses={404: {"Description": "Not found"}},
)

class Item(BaseModel):
    item: str

@router.post("/")
async def index(item: Item) -> str:
    val: str = item.item
    tensor: TensorFlow = TensorFlow()
    return tensor.init(image_64=val)


