from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class LocalLLM:
    def __init__(self, model_name: str, max_new_tokens: int):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
        )

    def generate(self, prompt: str) -> str:
        return self.pipe(prompt)[0]["generated_text"]
