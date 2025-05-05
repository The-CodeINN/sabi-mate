from fastapi import FastAPI

from ai_companion.interfaces.whatsapp.whatsapp_response import whatsapp_router

app = FastAPI()
app.include_router(
    whatsapp_router,
    prefix="/whatsapp",
    generate_unique_id_function=lambda route: route.name,
    tags=["whatsapp"],
)
