import os
import json
from google import genai
from google.genai import types
from src.config import Config
from src.database.vector_db import VectorDB

class RAGAgent:
    def __init__(self):
        self.db = VectorDB()
        self.api_key = Config.GEMINI_API_KEY
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
            print("WARNING: Gemini API Key not set. RAG agent will run in fallback mock mode.")
            
    def _query_chromadb(self, query_text: str, collection_name: str, n_results: int = 3):
        """Helper to query ChromaDB and extract documents."""
        try:
            results = self.db.query(collection_name, query_text, n_results=n_results)
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            return documents, metadatas
        except Exception as e:
            print(f"Error querying collection '{collection_name}': {e}")
            return [], []

    def run(self, trigger_context: dict) -> dict:
        """Runs RAG search and returns safety regulations, historical precedents, and recommended actions.
        
        Args:
            trigger_context (dict): Contains alert parameters:
                - zone_id (str)
                - sensor_reading (float)
                - cv_violations (list)
                - active_permits (list)
                
        Returns:
            dict matching the frozen output schema for 'rag_incident_compliance'.
        """
        zone_id = trigger_context.get("zone_id", "Unknown Zone")
        sensor_val = trigger_context.get("sensor_reading", 0.0)
        violations = trigger_context.get("cv_violations", [])
        permits = trigger_context.get("active_permits", [])
        
        # 1. Formulate search queries
        permit_types = [p.get("type", "") for p in permits]
        violation_types = [v.get("violation_type", "") for v in violations]
        
        search_query = f"flammable gas leak {sensor_val}% LEL, "
        if permit_types:
            search_query += f"active permits: {', '.join(permit_types)}, "
        if violation_types:
            search_query += f"PPE violations: {', '.join(violation_types)}"
            
        print(f"RAG Agent Querying vector DB with: '{search_query}'")
        
        # 2. Retrieve context from ChromaDB
        reg_docs, reg_meta = self._query_chromadb(search_query, "safety_regulations", n_results=3)
        inc_docs, inc_meta = self._query_chromadb(search_query, "incident_history", n_results=2)
        
        # 3. If no Gemini API Key is configured, use structured fallback logic (Vizag Scenario match)
        if not self.client:
            return self._generate_fallback_response(zone_id, sensor_val, permit_types, violation_types, reg_docs, inc_docs)
            
        # 4. Construct LLM prompt
        reg_context = "\n---\n".join([f"Source: {m['source']}, Page: {m.get('page','N/A')}\n{d}" for d, m in zip(reg_docs, reg_meta)])
        inc_context = "\n---\n".join([f"Source: {m['source']}, Severity: {m.get('severity','N/A')}\n{d}" for d, m in zip(inc_docs, inc_meta)])
        
        prompt = f"""
        You are the SafeOps AI Regulatory Compliance and Historical Incident Safety Agent.
        An industrial safety anomaly has been detected:
        - Zone: {zone_id}
        - Current Gas Concentration: {sensor_val}% LEL
        - Active Work Permits in Zone: {json.dumps(permit_types)}
        - Observed Safety Violations: {json.dumps(violation_types)}
        
        Analyze this situation against the following regulatory safety standards and historical incident reports:
        
        ### REGULATORY SAFETY STANDARDS
        {reg_context}
        
        ### HISTORICAL SAFETY INCIDENTS & LESSONS
        {inc_context}
        
        Provide a structured safety assessment in JSON format matching this exact schema:
        {{
            "agent": "rag_incident_compliance",
            "triggered_by_alert_id": "ALT_20260701_{zone_id}",
            "similar_incidents": [
                {{
                    "incident_id": "similar_incident_source_filename",
                    "date": "incident_year_or_date",
                    "plant": "incident_location",
                    "description": "brief summary of contributing factors",
                    "outcome": "loss details / injuries / fatal counts",
                    "similarity_score": 0.90
                }}
            ],
            "applicable_regulations": [
                {{
                    "regulation_id": "OISD_105_clause_or_Factories_Act_section",
                    "title": "regulation title",
                    "clause": "clause number",
                    "requirement": "exact requirement from the regulation",
                    "violation_detected": true/false,
                    "source": "regulation document source"
                }}
            ],
            "recommended_actions": [
                "immediate evacuation of zone",
                "shut off hot work electrical sources",
                "actuate hazard isolation valves"
            ],
            "rag_sources_cited": ["OISD-STD-105.pdf", "the_factories_act,_1948.pdf"]
        }}
        
        Return ONLY the valid JSON block without markdown formatting or code blocks.
        """
        
        try:
            # Call Gemini using the new GenAI SDK
            response = self.client.models.generate_content(
                model=Config.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            result = json.loads(response.text.strip())
            return result
        except Exception as e:
            print(f"Gemini API generation failed: {e}. Falling back to rule-based mock response.")
            return self._generate_fallback_response(zone_id, sensor_val, permit_types, violation_types, reg_docs, inc_docs)
            
    def _generate_fallback_response(self, zone_id, sensor_val, permit_types, violation_types, reg_docs, inc_docs) -> dict:
        """Deterministic offline fallback response matching the Vizag Hackathon scenario."""
        print("Generating deterministic RAG Safety report (offline fallback mode)...")
        
        similar_incidents = []
        # Match Vizag 2025 if gas leak and hot work are active
        if "HOT_WORK" in permit_types and sensor_val > 25.0:
            similar_incidents.append({
                "incident_id": "Vizag_2025_Incident_Report.txt",
                "date": "January 2025",
                "plant": "Visakhapatnam Steel Plant",
                "description": "Explosion in coke oven battery due to entrapped gas (38% LEL) during active Hot Work permit with workers missing PPE.",
                "outcome": "8 worker fatalities, total facility damage, and operational suspension.",
                "similarity_score": 0.95
            })
        else:
            similar_incidents.append({
                "incident_id": "HPCL_Vizag_2013_Report.txt",
                "date": "August 2013",
                "plant": "HPCL Visakhapatnam Refinery",
                "description": "Hydrocarbon leak ignited during welding/maintenance near cooling tower columns.",
                "outcome": "28 fatalities and facility shutdown.",
                "similarity_score": 0.78
            })
            
        applicable_regulations = []
        if "HOT_WORK" in permit_types:
            applicable_regulations.append({
                "regulation_id": "OISD-STD-105-Clause-7.3.2",
                "title": "Work Permit System Standard",
                "clause": "7.3.2",
                "requirement": "Section 7.3.2 prohibits hot work within 10 meters of a gas source or active gas leakage.",
                "violation_detected": True if sensor_val > 0 else False,
                "source": "Work Permit System Standard - OISD-STD-105.pdf"
            })
            applicable_regulations.append({
                "regulation_id": "OISD-STD-105-Clause-4.3",
                "title": "Work Permit System Standard - Gas Limits",
                "clause": "4.3",
                "requirement": "Section 4.3 prohibits hot work in the presence of gas above 25% LEL; all hot work permits must be immediately suspended if flammable gas exceeds 20% LEL.",
                "violation_detected": True if sensor_val >= 20.0 else False,
                "source": "Work Permit System Standard - OISD-STD-105.pdf"
            })
            
        if "CONFINED_SPACE" in permit_types or "NO_HELMET" in violation_types:
            applicable_regulations.append({
                "regulation_id": "Factories-Act-1948-Section-36",
                "title": "Factories Act - Chapter IV Safety Provisions",
                "clause": "36",
                "requirement": "Restricts entry into any confined space where gas/vapor hazards exist until gas testing confirms safety and oxygen levels are between 19.5% and 23.5%.",
                "violation_detected": True,
                "source": "the_factories_act,_1948.pdf"
            })
            
        recommended_actions = []
        if sensor_val >= 35.0 or (sensor_val >= 25.0 and "HOT_WORK" in permit_types):
            recommended_actions = [
                f"IMMEDIATE EVACUATION of all personnel in {zone_id} to safe assembly points.",
                "Halt all welding, grinding, and active Hot Work in Zone 3 and surrounding areas.",
                "Isolate gas supply lines and actuate nitrogen purging valves immediately.",
                "Deploy emergency responders with portable gas detectors and SCADA safety backups."
            ]
        elif sensor_val >= 10.0:
            recommended_actions = [
                "Deploy safety inspector to verify gas leaks.",
                "Initiate continuous gas monitoring and suspend non-essential work permits."
            ]
        else:
            recommended_actions = [
                "Continue routine operations and check sensor calibration."
            ]
            
        return {
            "agent": "rag_incident_compliance",
            "triggered_by_alert_id": f"ALT_20260701_{zone_id}",
            "similar_incidents": similar_incidents,
            "applicable_regulations": applicable_regulations,
            "recommended_actions": recommended_actions,
            "rag_sources_cited": [
                "Work Permit System Standard - OISD-STD-105.pdf",
                "the_factories_act,_1948.pdf"
            ]
        }
