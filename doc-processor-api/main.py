import hashlib
from typing import Annotated

from document_processing import JSON_Processing, PDF_to_Image, OCR_Model
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
app=FastAPI()
model=OCR_Model()
@app.get("/")
def status():
    return{"status":"running"}

@app.post("/document_processing")

async def document_processor(
    file: UploadFile = File(...),
    document_id: Annotated[str | None, Form()] = None,
    source_doc_uid: Annotated[str | None, Form()] = None,
):
    filename = file.filename or "document.pdf"
    source_doc_uid = (source_doc_uid.strip() or None) if source_doc_uid else None
    document_id = (document_id.strip() or None) if document_id else None
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400,detail="Insert a valid PDF file")
    
    pdf=await file.read()
    if not pdf:

        raise HTTPException(status_code=400,detail="Empty pdf")
    try:
        pdf_to_image=PDF_to_Image().convert(pdf)
        output=model(pdf_to_image)
       
        stable_document_id = (
            source_doc_uid
            or document_id
            or f"sha256-{hashlib.sha256(pdf).hexdigest()}"
        )
        return JSON_Processing(
            output,
            pdf_to_image,
            document_id=stable_document_id,
            source_doc_uid=source_doc_uid,
            original_filename=filename,
        )



    except Exception as e:
        raise HTTPException(status_code=400,detail=f"{e}")
