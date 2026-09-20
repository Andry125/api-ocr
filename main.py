import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from google import genai
from google.genai.errors import APIError
from PIL import Image
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

app = FastAPI(title="API OCR Gemini avec Auto-Retry")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# Fonction helper pour vérifier si l'erreur est temporaire (503 ou 429)
def is_transient_error(exception):
    if isinstance(exception, APIError):
        code = getattr(exception, "code", None)
        err_str = str(exception)
        # Re-essayer uniquement si c'est un problème de charge/quota temporaire (503 ou 429)
        return code in [503, 429] or "503" in err_str or "429" in err_str
    return False


# Décorateur de retry :
# - Réessaye jusqu'à 4 fois maximum
# - Attends de manière exponentielle (ex: 1s, 2s, 4s, 8s)
@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(is_transient_error),
    reraise=True,
)
def call_gemini_with_retry(model_name: str, contents: list):
    """Effectue l'appel à Gemini avec re-tentative automatique en cas de surcharge."""
    return client.models.generate_content(model=model_name, contents=contents)


@app.post("/ocr")
async def extract_text_from_image(file: UploadFile = File(...)):
    if not client:
        raise HTTPException(
            status_code=500,
            detail="La clé GEMINI_API_KEY n'est pas configurée.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Le fichier fourni doit être une image."
        )

    try:
        image = Image.open(file.file)
        prompt = "Extrais tout le texte visible dans cette image. Restitue uniquement le texte extrait sans aucun commentaire ni explication."

        # Modèle actif (ex: gemini-3.5-flash)
        model_name = "gemini-3.5-flash"

        # L'appel à la fonction décorée gère automatiquement les retries
        response = call_gemini_with_retry(
            model_name=model_name, contents=[prompt, image]
        )

        return {
            "filename": file.filename,
            "extracted_text": response.text.strip() if response.text else "",
        }

    except APIError as e:
        raise HTTPException(
            status_code=502,
            detail=f"L'API Gemini est toujours indisponible après plusieurs tentatives : {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur lors du traitement : {str(e)}"
        )
