from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    target_lang: str = Field(
        ..., description="Target language code (en, zh, ar, ko)", example="en"
    )
    text: str = Field(
        ...,
        description="Text to be translated",
        example="Saya suka belajar bahasa asing.",
    )


class TranslateHtmlRequest(BaseModel):
    target_lang: str = Field(
        ..., description="Target language code (en, zh, ar, ko)", example="ko"
    )
    html: str = Field(
        ..., description="HTML content to be translated", example="<p>Kucing</p>"
    )
