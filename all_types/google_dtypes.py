from pydantic import BaseModel


class SingleGoogleGeoDataElement(BaseModel):
    business_status: str
    formatted_address: str
    formatted_phone_number: str
    geometry: dict[str, dict]
    name: str
    opening_hours: dict[str, str]
    photos: list[dict[str, str]]
    rating: float
    user_ratings_total: int
    website: str


class GglResponse(BaseModel):
    list[SingleGoogleGeoDataElement]