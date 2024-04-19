from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class TextSummarizer:
    def __init__(self, model_name: str):
        """
        Initializes the TextSummarizer with a model specified by model_name.
        
        Args:
        model_name (str): The name of the model on Hugging Face's model hub.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def summarize(self, text: str, max_length: int = 150, num_beams: int = 4) -> str:
        """
        Summarizes the given text using the loaded model.
        
        Args:
        text (str): The text to be summarized.
        max_length (int): The maximum length of the summary.
        num_beams (int): The number of beams for beam search.

        Returns:
        str: The summarized text.
        """
        # Encode the text into tensor of ids
        encoded_input = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        # Generate the summary output
        with torch.no_grad():
            summarized_ids = self.model.generate(
                encoded_input["input_ids"],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )

        # Decode the tensor ids to text
        summarized_text = self.tokenizer.decode(summarized_ids[0], skip_special_tokens=True)
        return summarized_text