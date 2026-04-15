import os
import pdfplumber
import operator
from typing import Optional, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))
hf_token = os.getenv("HUGGINGFACE_API_KEY")

structuring_llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=1024,
    temperature=0.1, 
    huggingfacehub_api_token=hf_token
)
structuring_chat_model = ChatHuggingFace(llm=structuring_llm)

class InsurancePolicy(BaseModel):
    policy_type: Optional[str] = Field(None, description="Type of insurance (e.g., 'Life', 'Health', 'Car')")
    premium_amount: Optional[float] = Field(None, description="Premium amount paid")
    payment_date: Optional[str] = Field(None, description="Date the premium was paid")

class BankAccountData(BaseModel):
    bank_name: Optional[str] = Field(None, description="Name of the bank or last 4 digits of the account number")
    interest_income: Optional[float] = Field(None, description="Total interest earned or credited in this account")
    high_value_transactions: Optional[list[float]] = Field(default_factory=list, description="List of transactions exceeding 50,000 INR")

class TaxDocumentData(BaseModel):
    employer_tan: Optional[str] = Field(None, description="Employer TAN (Tax Deduction Account Number)")
    employee_pan: Optional[str] = Field(None, description="Employee PAN (Permanent Account Number)")
    gross_salary: Optional[float] = Field(None, description="Total gross salary before deductions")
    net_salary: Optional[float] = Field(None, description="Net salary after deductions")
    exemptions_10: Optional[float] = Field(None, description="Section 10 exemptions like HRA, LTA")
    chapter_via_deductions: Optional[float] = Field(None, description="80C, 80D, 80G deductions")
    total_tds_deducted: Optional[float] = Field(None, description="Total Tax Deducted at Source")
    net_tax_payable: Optional[float] = Field(None, description="Net tax payable amount")
    principal_repayment: Optional[float] = Field(None, description="Principal amount repaid (80C)")
    interest_repayment: Optional[float] = Field(None, description="Interest amount repaid (Sec 24 or 80E)")
    bank_accounts: Optional[list[BankAccountData]] = Field(default_factory=list, description="List of bank accounts")
    insurance_policies: Optional[list[InsurancePolicy]] = Field(default_factory=list, description="List of insurance policies")

class OverallState(TypedDict):
    file_paths: list[str]       
    parsed_documents: Annotated[list[TaxDocumentData], operator.add] 
    final_extracted_data: Optional[TaxDocumentData]

class DocumentState(TypedDict):
    file_path: str

def map_documents(state: OverallState):
    print(f"Mapping {len(state['file_paths'])} documents for parallel processing...")
    return [Send("process_single_document", {"file_path": path}) for path in state["file_paths"]]

def process_single_document(state: DocumentState):
    file_path = state["file_path"]
    filename = os.path.basename(file_path)
    print(f"Processing isolated document: {filename}")
    
    raw_text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(layout=True)
                
                if not text or not text.strip():
                    text = page.extract_text()
                    
                if text: 
                    raw_text += text + "\n"
    except Exception as e:
        print(f"PDF Error on {filename}: {e}")
        return {"parsed_documents": [TaxDocumentData()]}

    if not raw_text.strip():
        print(f"No readable text found in {filename} (Likely a scanned image). Skipping LLM.")
        return {"parsed_documents": [TaxDocumentData()]}

    parser = PydanticOutputParser(pydantic_object=TaxDocumentData)
    format_instructions = parser.get_format_instructions()
    
    try:
        prompt = f"""
        You are a highly precise financial data extraction engine. 
        Extract the relevant entities from this single tax document into the requested schema.
        If a value is not explicitly present in the text, you MUST leave it as null. Do not guess.
        
        {format_instructions}
        
        RAW DOCUMENT TEXT ({filename}):
        {raw_text}
        """
        response = structuring_chat_model.invoke([HumanMessage(content=prompt)])
        structured_result = parser.invoke(response.content)
    except Exception as e:
        print(f"LLM Structuring Error on {filename}: {e}")
        structured_result = TaxDocumentData()
        
    return {"parsed_documents": [structured_result]}

def reduce_documents(state: OverallState):
    print("Reducing and merging all extracted data...")
    master_profile = TaxDocumentData()
    
    for doc in state["parsed_documents"]:
        for field in TaxDocumentData.model_fields:
            if field not in ['bank_accounts', 'insurance_policies']:
                current_val = getattr(master_profile, field)
                new_val = getattr(doc, field)
                if current_val is None and new_val is not None:
                    setattr(master_profile, field, new_val)
        
        if doc.bank_accounts:
            master_profile.bank_accounts.extend(doc.bank_accounts)
        if doc.insurance_policies:
            master_profile.insurance_policies.extend(doc.insurance_policies)
            
    return {"final_extracted_data": master_profile}

ocr_builder = StateGraph(OverallState)

ocr_builder.add_node("process_single_document", process_single_document)
ocr_builder.add_node("reduce_documents", reduce_documents)

ocr_builder.add_conditional_edges(START, map_documents)

ocr_builder.add_edge("process_single_document", "reduce_documents")
ocr_builder.add_edge("reduce_documents", END)

ocr_subgraph = ocr_builder.compile()

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    
    TEST_FILES = [
        os.path.join(current_dir, "sample_form16.pdf"), 
        os.path.join(current_dir, "bank_statement_3.pdf"),
        os.path.join(current_dir, "receipt.pdf") 
    ]
    
    valid_files = [f for f in TEST_FILES if os.path.exists(f)]
    
    if not valid_files:
        print("Could not find test files.")
    else:
        print("\nSTARTING MAP-REDUCE OCR AGENT")
        initial_state = {"file_paths": valid_files}
        final_state = ocr_subgraph.invoke(initial_state)
        
        print("\nFINAL MERGED JSON PROFILE")
        extracted_data = final_state.get("final_extracted_data")
        
        if extracted_data:
            print(extracted_data.model_dump_json(indent=4))
        else:
            print("Failed to extract data.")