from document_processing import JSON_Processing,PDF_to_Image,OCR_Model,Table_Model
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
        pdf_to_image, png_bytes_list = PDF_to_Image().convert(pdf)
        ocr_output = OCR_Model()(pdf_to_image)
        table_model = Table_Model()
        return JSON_Processing(ocr_output, pdf_to_image, png_bytes_list, pdf, table_model)



    except Exception as e:
        raise HTTPException(status_code=400,detail=f"{e}")


