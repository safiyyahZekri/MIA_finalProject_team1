from doctr.models import ocr_predictor
from doctr.io import DocumentFile 
import uuid
from typing import List,Union,Literal
from pydantic import BaseModel
import numpy as np
import pymupdf

class Word(BaseModel):
    word_list:List[str]
    bbox_list:List[List[int]]


class Block(BaseModel):
    bbox:List[int]
    uuid:str
    words:Word
    text:str
    order:int
    content_type:Literal["paragraph"]

class Cell(BaseModel):
    bbox:List[int]
    text:str
    row_span:List[int]
    col_span:List[int]

class Table_Block(BaseModel):
    bbox:List[int]
    uuid:str
    text:str
    order:int
    content_type:Literal["table"]
    cells:List[Cell]
   
    


class Page(BaseModel):
    bbox:List[int]
    blocks:List[Union[Block,Table_Block]]
    page_number:int

class DocumentJSON(BaseModel):
    pages:List[Page]
    document_id:str

class PDF_to_Image():
    def convert(self,pdf_path):
        images=[]
        image=pymupdf.open(stream=pdf_path,filetype="pdf")
        page_count=image.page_count
        for page in range(page_count):
            image_bytes=image[page].get_pixmap(matrix=pymupdf.Matrix(2,2),alpha=False,colorspace=pymupdf.csRGB)
            image_bytes_flattened=image_bytes.samples
            image_matrix=np.frombuffer(image_bytes_flattened,dtype=np.uint8).reshape(image_bytes.height,image_bytes.width,3)
            images.append(image_matrix)
        return images
            

class OCR_Model():
    def __init__(self):
        self.model=ocr_predictor(pretrained=True,detect_tables=True,detect_layout=True)
    def __call__(self,pdf_to_image):
        return self.model(pdf_to_image)




def JSON_Processing(output,pdf_to_image):
    pages_list=[]
    for page_number,page in enumerate(output.pages):
        page_no=page_number+1
        height,width=pdf_to_image[page_number].shape[:2]
        blocks_list=[]
        for block in page.blocks:
            for line in block.lines:
                x2=y2=100000
                x3=y3=-1 #Line geometry isnt provided so we use the min and max geometry of each word
                words_list=[]
                bbox_list=[]
                for word in line.words:
                    (x0,y0),(x1,y1)=word.geometry
                    x0*=int(width)
                    x1*=int(width)
                    y0*=int(height)
                    y1*=int(height)
                    x2=min(x2,x0)
                    y2=min(y2,y0)
                    x3=max(x3,x1)
                    y3=max(y3,y1)
                    word_text=word.value
                    words_list.append(word_text)
                    bbox_list.append([int(x0),int(y0),int(x1),int(y1)])
                words=Word(word_list=words_list,bbox_list=bbox_list)
                text=" ".join(words.word_list)
                uuid_=str(uuid.uuid4())
                order=len(blocks_list)+1        
                bbox=[int(x2),int(y2),int(x3),int(y3)]
                content_type="paragraph"
                blocks=Block(bbox=bbox,uuid=uuid_,words=words,text=text,order=order,content_type=content_type)
                blocks_list.append(blocks)
        for table in page.tables:
            cells=table.cells

            cells=sorted(cells,key=lambda cell:(cell.row_start,cell.col_start,cell.row_end,cell.col_end,cell.col_end))
            cells_list=[]
            text_list=[]
            for cell in cells:
                (x0,y0),(x1,y1)=cell.geometry
                x0*=int(width)
                x1*=int(width)
                y0*=int(height)
                y1*=int(height)
                bbox=[int(x0),int(y0),int(x1),int(y1)]
        
                text=cell.value
                row_span=[cell.row_start,cell.row_end]
                col_span=[cell.col_start,cell.col_end]
                text_list.append(cell.value)
                cell=Cell(bbox=[int(x0),int(y0),int(x1),int(y1)],text=cell.value,row_span=row_span,col_span=col_span)
                cells_list.append(cell)
            table_text=" ".join(text_list)
            uuid_=str(uuid.uuid4())
            order=len(blocks_list)+1
            content_type="table"
            (x0,y0),(x1,y1)=table.geometry
            x0*=int(width)
            x1*=int(width)
            y0*=int(height)
            y1*=int(height)
            block=Table_Block(bbox=[int(x0),int(y0),int(x1),int(y1)],content_type=content_type,order=order,text=table_text,uuid=uuid_,cells=cells_list)
            blocks_list.append(block)  
        blocks_list=sorted(blocks_list,key=lambda block:(block.bbox[1],block.bbox[0])) 
        for idx,block in enumerate(blocks_list):
            block.order=idx+1
        page_block=Page(bbox=[0,0,int(width),int(height)],page_number=page_no,blocks=blocks_list)
        pages_list.append(page_block)
                        
    return DocumentJSON(pages=pages_list,document_id=str(uuid.uuid4()))
                
    

                    
                    
        