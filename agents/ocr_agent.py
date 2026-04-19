import os
import io
import operator
import fitz
import pytesseract
from PIL import Image, ImageFilter
from typing import Optional, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq 
from langchain_ollama import ChatOllama
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(project_root, ".env"))

groq_api_key = os.getenv("GROQ_API_KEY")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

structuring_chat_model = ChatGroq(
    
model="llama-3.1-8b-instant",
    temperature=0.1,
   groq_api_key=groq_api_key,
   max_retries=5
)

class InsurancePolicy(BaseModel):
    policy_type: Optional[str] = Field(None)
    premium_amount: Optional[float] = Field(None)
    payment_date: Optional[str] = Field(None)

class BankAccountData(BaseModel):
    bank_name: Optional[str] = Field(None)
    interest_income: Optional[float] = Field(None)
    high_value_transactions: Optional[list[float]] = Field(default_factory=list)

class TaxDocumentData(BaseModel):
    employer_tan: Optional[str] = Field(None)
    employee_pan: Optional[str] = Field(None)
    gross_salary: Optional[float] = Field(None)
    net_salary: Optional[float] = Field(None)
    exemptions_10: Optional[float] = Field(None)
    chapter_via_deductions: Optional[float] = Field(None)
    total_tds_deducted: Optional[float] = Field(None)
    net_tax_payable: Optional[float] = Field(None)
    principal_repayment: Optional[float] = Field(None)
    interest_repayment: Optional[float] = Field(None)
    bank_accounts: Optional[list[BankAccountData]] = Field(default_factory=list)
    insurance_policies: Optional[list[InsurancePolicy]] = Field(default_factory=list)

class OverallState(TypedDict):
    file_paths: list[str]
    parsed_documents: Annotated[list[TaxDocumentData], operator.add]
    final_extracted_data: Optional[TaxDocumentData]

class DocumentState(TypedDict):
    file_path: str

def map_documents(state: OverallState):
    return [Send("process_single_document", {"file_path": path}) for path in state["file_paths"]]

def process_single_document(state: DocumentState):
    file_path = state["file_path"]
    filename = os.path.basename(file_path)
    raw_text = ""

    
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text = page.get_text()
            if text.strip():
                raw_text += text + "\n"
    except Exception as e:
        print(f"PDF Extraction Error on {filename}: {e}")

   
    if not raw_text.strip():
        print(f"Invisible text detected in {filename}. Running Tesseract OCR...")
        try:
            doc = fitz.open(file_path)
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
                
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                img = img.convert("L")
                img = img.filter(ImageFilter.SHARPEN)
                
                text = pytesseract.image_to_string(img, config="--psm 6")
                if text.strip():
                    raw_text += text + "\n"
        except Exception as e:
            print(f"Tesseract OCR Error on {filename}: {e}")
            return {"parsed_documents": [TaxDocumentData()]}

   
    structured_llm = structuring_chat_model.with_structured_output(TaxDocumentData)

    try:
     
        prompt = f"""
        You are a highly precise financial data extraction engine.
        Extract the relevant entities from this single tax document into the requested schema.
        If a value is not explicitly present in the text, leave it as null.
        If a document does not contain bank accounts or insurance policies, return an empty list [].

        RAW DOCUMENT TEXT ({filename}):
        {raw_text}
        """
        
        structured_result = structured_llm.invoke([HumanMessage(content=prompt)])
        
    except Exception as e:
        print(f"LLM Structuring Error on {filename}: {e}")
        structured_result = TaxDocumentData()

    return {"parsed_documents": [structured_result]}

def reduce_documents(state: OverallState):
    master_profile = TaxDocumentData()
    for doc in state["parsed_documents"]:
        for field in TaxDocumentData.model_fields:
            if field not in ['bank_accounts', 'insurance_policies']:
                current_val = getattr(master_profile, field)
                new_val = getattr(doc, field)
                if current_val is None and new_val is not None:
                    setattr(master_profile, field, new_val)

        if doc.bank_accounts:
            valid_banks = [
                b for b in doc.bank_accounts
                if b.bank_name is not None or b.interest_income is not None or b.high_value_transactions
            ]
            master_profile.bank_accounts.extend(valid_banks)

        if doc.insurance_policies:
            valid_policies = [
                p for p in doc.insurance_policies
                if p.policy_type is not None or p.premium_amount is not None or p.payment_date is not None
            ]
            master_profile.insurance_policies.extend(valid_policies)

    return {"final_extracted_data": master_profile}

ocr_builder = StateGraph(OverallState)
ocr_builder.add_node("process_single_document", process_single_document)
ocr_builder.add_node("reduce_documents", reduce_documents)
ocr_builder.add_conditional_edges(START, map_documents)
ocr_builder.add_edge("process_single_document", "reduce_documents")
ocr_builder.add_edge("reduce_documents", END)
ocr_subgraph = ocr_builder.compile()

if __name__ == "__main__":
    data_dir = os.path.join(project_root, "data")
    TEST_FILES = [
        os.path.join(data_dir, "sample_form16.pdf"),
        os.path.join(data_dir, "sample_insurance_Receipt.pdf-compressed.pdf"),
        os.path.join(data_dir, "dummy_statement-compressed.pdf")
    ]

    valid_files = [f for f in TEST_FILES if os.path.exists(f)]

    if valid_files:
        initial_state = {"file_paths": valid_files}
        final_state = ocr_subgraph.invoke(initial_state)
        extracted_data = final_state.get("final_extracted_data")

        if extracted_data:
            print(extracted_data.model_dump_json(indent=4))