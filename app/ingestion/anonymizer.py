from typing import Dict, Tuple
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

class GDPRAnonymizer:
    def __init__(self):
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "en", "model_name": "en_core_web_sm"},
                {"lang_code": "de", "model_name": "de_core_news_sm"},
            ],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=["en", "de"],
        )
        self.anonymizer = AnonymizerEngine()
        self._register_compliance_patterns()

    def _register_compliance_patterns(self):
        iban_pattern = Pattern(
            name="eu_iban_regex",
            regex=r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
            score=0.95,
        )
        for language in ("en", "de"):
            self.analyzer.registry.add_recognizer(
                PatternRecognizer(
                    supported_entity="IBAN",
                    patterns=[iban_pattern],
                    supported_language=language,
                )
            )

        tax_pattern = Pattern(
            name="de_tax_id_regex",
            regex=r"\b\d{2}\s?\d{3}\s?\d{3}\s?\d{3}\b",
            score=0.90,
        )
        self.analyzer.registry.add_recognizer(
            PatternRecognizer(
                supported_entity="DE_TAX_ID",
                patterns=[tax_pattern],
                supported_language="de",
            )
        )

    def mask_pii(self, text: str, language: str = "en") -> Tuple[str, Dict[str, str]]:
        results = self.analyzer.analyze(
            text=text,
            language=language,
            entities=[
                "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
                "IBAN", "DE_TAX_ID", "IP_ADDRESS",
            ],
        )
        vault: Dict[str, str] = {}
        counters: Dict[str, int] = {}
        anonymized_text = text

        for result in sorted(results, key=lambda item: item.start, reverse=True):
            raw_value = text[result.start:result.end]
            entity = result.entity_type
            index = counters.get(entity, 1)
            token = f"<{entity}_{index}>"
            counters[entity] = index + 1
            vault[token] = raw_value
            anonymized_text = (
                anonymized_text[:result.start] + token + anonymized_text[result.end:]
            )
        return anonymized_text, vault

    def deanonymize(self, text: str, vault: Dict[str, str]) -> str:
        restored = text
        for token, original in vault.items():
            restored = restored.replace(token, original)
        return restored
