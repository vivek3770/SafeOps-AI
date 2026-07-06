import os
import sys
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.utils.doc_parser import extract_text_from_pdf, chunk_text
from src.database.vector_db import VectorDB

# Detailed historical incident datasets to seed
HISTORICAL_INCIDENTS = [
    {
        "text": """Visakhapatnam Steel Plant Coke Oven Battery Explosion (January 2025).
        An explosion occurred in the coke oven battery facility resulting in 8 worker fatalities.
        Contributing Factors:
        1. Entrapped toxic and flammable gases (carbon monoxide and methane) triggered a sudden explosion.
        2. Functional safety systems (gas detectors and SCADA) showed elevated readings up to 38% LEL, but no intelligence layer integrated these readings with operational permit logs in real time.
        3. A Hot Work permit was actively issued in the immediate proximity of the battery area while gas pressure readings were abnormally rising.
        4. Multiple workers were present in the hazard zone without adequate PPE (missing hardhats and protective vests).
        Regulatory Violation: OISD Standard 105 Section 4.3 prohibiting hot work in presence of gas above 25% LEL.""",
        "metadata": {"source": "Vizag_2025_Incident_Report.txt", "type": "incident", "severity": "CRITICAL"}
    },
    {
        "text": """HPCL Visakhapatnam Refinery Cooling Tower Fire (August 2013).
        A massive fire broke out at the cooling tower of the refinery, resulting in 28 fatalities.
        Contributing Factors:
        1. Hydrocarbon gas leakage from an under-maintenance pipeline occurred near the cooling tower area.
        2. Maintenance work (welding/hot work) was being performed on the cooling tower columns simultaneously.
        3. The permit-to-work system did not check for overlapping gas readings or enforce strict gas testing protocols before welding.
        4. Lack of immediate gas alarms and delay in evacuation protocols led to high casualties.
        Regulatory Violation: Factories Act 1948 Chapter IV Section 36 on confined spaces and gas containment.""",
        "metadata": {"source": "HPCL_Vizag_2013_Report.txt", "type": "incident", "severity": "CRITICAL"}
    },
    {
        "text": """IOCL Jaipur Terminal Oil Depot Fire (October 2009).
        A major fire and series of explosions occurred at the Indian Oil Corporation terminal at Sitapura, Jaipur, killing 12 people.
        Contributing Factors:
        1. Massive leakage of petrol/gasoline during transfer operations from storage tanks.
        2. Formation of a large vapor cloud due to lack of immediate isolation of the leak.
        3. An active electrical spark/maintenance activity in the adjacent zone acted as the ignition source.
        4. The permit control failed to halt all active hot work and electrical work in adjacent zones upon detection of major fuel leakage.
        Regulatory Violation: OISD Standard 105 on Emergency Isolation and Permit Suspension.""",
        "metadata": {"source": "IOC_Jaipur_2009_Report.txt", "type": "incident", "severity": "CRITICAL"}
    }
]

# Synthetic near-miss reports
NEAR_MISS_REPORTS = [
    {
        "text": "Near-Miss: Gas alarm flagged at 30% LEL in Zone-2 during flange tightening. Hot work permit active in Zone-3 (adjacent). Hot work was suspended immediately by safety officer. No ignition occurred.",
        "metadata": {"source": "Near_Miss_Log_001.txt", "type": "near_miss", "severity": "HIGH"}
    },
    {
        "text": "Near-Miss: Confined space entry permit active in Zone-1 (Vessel V-102) while ventilation fan failed for 10 minutes. Oxygen levels dropped to 19.1%. Evacuation alarm triggered and workers evacuated safely.",
        "metadata": {"source": "Near_Miss_Log_002.txt", "type": "near_miss", "severity": "HIGH"}
    },
    {
        "text": "Near-Miss: Scaffolding work (height work) active in Zone-5 without safety harness hooks secured. Safety inspector stopped the work and corrected PPE compliance before allowing resumption.",
        "metadata": {"source": "Near_Miss_Log_003.txt", "type": "near_miss", "severity": "MED"}
    }
]

# Permit templates/rules
PERMIT_TEMPLATES = [
    {
        "text": "PTW Hot Work Permit Guidelines: Requires pre-work gas testing. LEL must be below 10% for continuous work. If LEL reaches 20%, work must be suspended immediately. Requires fire watch, portable extinguishers, and safety barrier.",
        "metadata": {"source": "PTW_HotWork_Template.txt", "type": "permit_guideline"}
    },
    {
        "text": "PTW Confined Space Entry Permit Guidelines: Requires continuous ventilation, oxygen level monitoring (19.5% to 23.5% required), toxic gas testing (H2S < 10 ppm, CO < 30 ppm), and a standby watch at the entry point.",
        "metadata": {"source": "PTW_ConfinedSpace_Template.txt", "type": "permit_guideline"}
    }
]

def main():
    print("--- Starting ChromaDB Vector Database Seeding Script ---")
    db = VectorDB()
    
    # 1. Parse and seed regulatory PDFs if they exist
    pdf_sources = [
        ("Work Permit System Standard - OISD-STD-105.pdf", "safety_regulations"),
        ("the_factories_act,_1948.pdf", "safety_regulations")
    ]
    
    for filename, collection_name in pdf_sources:
        pdf_path = os.path.join(Config.RAW_DOCS_DIR, filename)
        if os.path.exists(pdf_path):
            print(f"Found PDF file: {filename}. Extracting and chunking...")
            try:
                pages = extract_text_from_pdf(pdf_path)
                if not pages:
                    print(f"No text extracted from PDF {filename} (scanned PDF). Loading fallback text...")
                    fallback_filename = filename.replace(".pdf", "_Fallback.txt").replace(" - ", "_").replace("-", "_").replace(" ", "_")
                    fallback_path = os.path.join(Config.RAW_DOCS_DIR, fallback_filename)
                    if os.path.exists(fallback_path):
                        with open(fallback_path, "r", encoding="utf-8") as f:
                            text = f.read()
                        # Split by double newline to separate clauses
                        sections = text.split("\n\n")
                        pages = [{
                            'text': sec.strip(),
                            'page': idx + 1,
                            'source': fallback_filename
                        } for idx, sec in enumerate(sections) if sec.strip()]
                    else:
                        print(f"Fallback file not found at: {fallback_path}")
                
                if pages:
                    chunks = chunk_text(pages, chunk_size_words=350, overlap_words=40)
                    documents = [c['text'] for c in chunks]
                    metadatas = [c['metadata'] for c in chunks]
                    ids = [c['metadata']['doc_id'] for c in chunks]
                    
                    print(f"Seeding {len(documents)} chunks from {filename} to '{collection_name}' collection...")
                    db.add_documents(collection_name, documents, metadatas, ids)
                    print(f"Successfully seeded {filename}.")
                else:
                    print(f"No source text available for {filename}. Skipping.")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"PDF file not found at: {pdf_path}. Skipping PDF parsing.")

    # 2. Seed incident history collection
    print("Seeding historical incident summaries to 'incident_history' collection...")
    incident_docs = [item['text'] for item in HISTORICAL_INCIDENTS]
    incident_metadatas = [item['metadata'] for item in HISTORICAL_INCIDENTS]
    incident_ids = [f"inc_{i}" for i in range(len(HISTORICAL_INCIDENTS))]
    
    db.add_documents("incident_history", incident_docs, incident_metadatas, incident_ids)
    print("Successfully seeded incident history.")

    # 3. Seed near-miss reports
    print("Seeding near-miss logs to 'incident_history' collection...")
    nm_docs = [item['text'] for item in NEAR_MISS_REPORTS]
    nm_metadatas = [item['metadata'] for item in NEAR_MISS_REPORTS]
    nm_ids = [f"nm_{i}" for i in range(len(NEAR_MISS_REPORTS))]
    
    db.add_documents("incident_history", nm_docs, nm_metadatas, nm_ids)
    print("Successfully seeded near-miss logs.")

    # 4. Seed permit templates
    print("Seeding permit guidelines to 'permit_templates' collection...")
    pt_docs = [item['text'] for item in PERMIT_TEMPLATES]
    pt_metadatas = [item['metadata'] for item in PERMIT_TEMPLATES]
    pt_ids = [f"pt_{i}" for i in range(len(PERMIT_TEMPLATES))]
    
    db.add_documents("permit_templates", pt_docs, pt_metadatas, pt_ids)
    print("Successfully seeded permit templates.")

    print("--- Database Seeding Complete ---")

if __name__ == "__main__":
    main()
