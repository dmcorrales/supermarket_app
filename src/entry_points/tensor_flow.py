from fastapi import APIRouter
router = APIRouter(
    prefix="/train",
    responses={404: {"Description": "Not found"}},
)


@router.get("/")
async def index() -> str:
    return "HOLA!"
