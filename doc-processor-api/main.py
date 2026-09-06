from document_processing import JSON_Processing,PDF_to_Image,OCR_Model
from fastapi import FastAPI,File,UploadFile,HTTPException
app=FastAPI()
model=OCR_Model()
@app.get("/")
def status():
    return{"status":"running"}

@app.post("/document_processing")

async def document_processor(file:UploadFile=File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400,detail="Insert a valid PDF file")
    
    pdf=await file.read()
    if not pdf:

        raise HTTPException(status_code=400,detail="Empty pdf")
    try:
        pdf_to_image=PDF_to_Image().convert(pdf)
        output=model(pdf_to_image)
       
        return JSON_Processing(output, pdf_to_image)



    except Exception as e:
        raise HTTPException(status_code=400,detail=f"{e}")


