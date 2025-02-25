import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

class SeparateEmotionAnalyzer:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.logger = self._setup_logger()
        self.prompts = self._load_prompts()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"emotion_analysis_{timestamp}.log"

        logger = logging.getLogger("separate_analyzer")
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _load_prompts(self) -> Dict[str, str]:
        """Load all prompt templates."""
        prompts = {}
        prompt_files = {
            'emotion': 'EmotionDetection.txt',
            'topic': 'TopicAnalysis.txt',
            'adorescore': 'Adorescore.txt',
            'output': 'OutputGenerator.txt'
        }

        for key, filename in prompt_files.items():
            path = Path(__file__).parent / "prompts" / filename
            try:
                with open(path, "r", encoding="utf-8") as f:
                    prompts[key] = f.read()
                    self.logger.info(f"Loaded {key} prompt from {path}")
            except FileNotFoundError:
                self.logger.error(f"Prompt file not found: {path}")
                raise FileNotFoundError(f"Prompt file not found: {path}")

        return prompts

    def _call_api(self, prompt: str, text: str) -> Optional[Dict]:
        """Make an API call to Gemini."""
        try:
            self.logger.info("Sending request to Gemini API")
            response = self.client.chat.completions.create(
                model="gemini-2.0-flash",
                n=1,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}
                ]
            )
            
            result = response.choices[0].message.content
            self.logger.info("Received response from Gemini API")
            
            # Clean up the response
            result = result.replace('```json', '').replace('```', '').strip()
            
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                self.logger.error(f"Error parsing JSON response: {result}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error calling Gemini API: {e}")
            return None

    def analyze_emotions(self, text: str) -> Optional[Dict]:
        """Analyze emotions in the text."""
        self.logger.info("Analyzing emotions")
        return self._call_api(self.prompts['emotion'], text)

    def analyze_topics(self, text: str) -> Optional[Dict]:
        """Analyze topics in the text."""
        self.logger.info("Analyzing topics")
        return self._call_api(self.prompts['topic'], text)

    def calculate_adorescore(self, text: str, topic_result: Dict) -> Optional[Dict]:
        """Calculate adorescores using both text and topic information."""
        self.logger.info("Calculating adorescores")
        
        # Combine the text and topic information for the API call
        combined_input = json.dumps({
            "text": text,
            "topics": topic_result
        })
        
        return self._call_api(self.prompts['adorescore'], combined_input)

    def generate_final_output(self, emotion_result: Dict, topic_result: Dict, 
                            adorescore_result: Dict) -> Optional[Dict]:
        """Combine results into final output format."""
        self.logger.info("Generating final output")
        
        # Prepare combined results as a string
        combined_input = json.dumps({
            "emotion_analysis": emotion_result,
            "topic_analysis": topic_result,
            "adorescore_analysis": adorescore_result
        })
        
        return self._call_api(self.prompts['output'], combined_input)

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

    def process_input_folder(self, input_folder: str, output_folder: str) -> None:
        """Process all text files in the input folder and save results."""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Processing input folder: {input_folder}")
        self.logger.info(f"Output folder: {output_folder}")

        for file_path in input_path.glob("*.txt"):
            self.logger.info(f"Processing file: {file_path.name}")
            
            # Read input text
            text = self._read_input_file(str(file_path))
            if not text:
                self.logger.warning(f"Skipping {file_path.name} due to read error")
                continue

            # Perform separate analyses
            emotion_result = self.analyze_emotions(text)
            if not emotion_result:
                self.logger.warning(f"Emotion analysis failed for {file_path.name}")
                continue

            topic_result = self.analyze_topics(text)
            if not topic_result:
                self.logger.warning(f"Topic analysis failed for {file_path.name}")
                continue

            # Calculate adorescore using both text and topics
            adorescore_result = self.calculate_adorescore(text, topic_result)
            if not adorescore_result:
                self.logger.warning(f"Adorescore calculation failed for {file_path.name}")
                continue

            # Generate final output
            final_result = self.generate_final_output(
                emotion_result, topic_result, adorescore_result
            )
            if not final_result:
                self.logger.warning(f"Final output generation failed for {file_path.name}")
                continue

            # Save result
            output_file = output_path / f"{file_path.stem}.json"
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(final_result, f, indent=2)
                self.logger.info(f"Analysis saved to {output_file}")
            except Exception as e:
                self.logger.error(f"Error saving results to {output_file}: {e}")

def main():
    # Initialize the analyzer
    analyzer = SeparateEmotionAnalyzer()

    # Define input and output folders
    input_folder = "inputs"
    output_folder = "./SeparateApproach/output"

    # Process all files
    analyzer.process_input_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()
