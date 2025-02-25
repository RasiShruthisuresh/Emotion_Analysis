import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

class EmotionAnalyzer:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.logger = self._setup_logger()
        self.system_prompt = self._load_system_prompt()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        # Create logs directory
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create a unique log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"emotion_analysis_{timestamp}.log"

        # Create logger
        logger = logging.getLogger("emotion_analyzer")
        logger.setLevel(logging.INFO)

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _load_system_prompt(self) -> str:
        """Load the system prompt from the file."""
        prompt_path = Path(__file__).parent / "prompt" / "system_prompt.txt"
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.logger.info(f"Loading system prompt from {prompt_path}")
                return f.read()
        except FileNotFoundError:
            self.logger.error(f"System prompt file not found at {prompt_path}")
            raise FileNotFoundError(f"System prompt file not found at {prompt_path}")

    def _read_input_file(self, file_path: str) -> Optional[str]:
        """Read content from an input file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.logger.info(f"Reading input file: {file_path}")
                return f.read().strip()
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return None

    def analyze_text(self, text: str) -> Optional[Dict]:
        """Analyze text using the Gemini API."""
        try:
            self.logger.info("Sending request to Gemini API")
            response = self.client.chat.completions.create(
                model="gemini-2.0-flash",
                n=1,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ]
            )
            
            # Extract the response content
            result = response.choices[0].message.content
            self.logger.info("Received response from Gemini API")
            
            # Clean up the response by removing markdown code block markers
            result = result.replace('```json', '').replace('```', '').strip()
            
            # Parse the JSON response
            try:
                parsed_result = json.loads(result)
                self.logger.info("Successfully parsed JSON response")
                return parsed_result
            except json.JSONDecodeError:
                self.logger.error(f"Error parsing JSON response: {result}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error calling Gemini API: {e}")
            return None

    def process_input_folder(self, input_folder: str, output_folder: str) -> None:
        """Process all text files in the input folder and save results."""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Processing input folder: {input_folder}")
        self.logger.info(f"Output folder: {output_folder}")

        # Process each .txt file in the input folder
        for file_path in input_path.glob("*.txt"):
            self.logger.info(f"Processing file: {file_path.name}")
            
            # Read input text
            text = self._read_input_file(str(file_path))
            if not text:
                self.logger.warning(f"Skipping {file_path.name} due to read error")
                continue

            # Analyze text
            result = self.analyze_text(text)
            if not result:
                self.logger.warning(f"Skipping {file_path.name} due to analysis error")
                continue

            # Save result
            output_file = output_path / f"{file_path.stem}.json"
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                self.logger.info(f"Analysis saved to {output_file}")
            except Exception as e:
                self.logger.error(f"Error saving results to {output_file}: {e}")

def main():
    # Initialize the analyzer
    analyzer = EmotionAnalyzer()

    # Define input and output folders
    input_folder = "inputs"
    output_folder = "./SinglePromptApproach/output"

    # Process all files
    analyzer.process_input_folder(input_folder, output_folder)

if __name__ == "__main__":
    main() 
