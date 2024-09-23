from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline

# Load a pre-trained model from Hugging Face's model hub (e.g., RoBERTa fine-tuned for AI detection)
model_name = "roberta-base"  # You can replace this with a fine-tuned model for text detection
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a text classification pipeline
ai_detector = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# Function to read the text from a file
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Truncate the text if it's too long for the model
def truncate_text(text, max_length=512):
    tokenized_text = tokenizer.encode(text, truncation=True, max_length=max_length)
    return tokenizer.decode(tokenized_text, skip_special_tokens=True)

# AI detection function with better output
def detect_ai(text):
    # Run the AI detector on the input text
    result = ai_detector(text)
    
    # Retrieve the scores
    human_score = result[0][0]['score']
    ai_score = result[0][1]['score']
    
    # Determine the final label
    if ai_score > human_score:
        final_label = "AI-generated"
    else:
        final_label = "Human-written"
    
    return final_label, human_score, ai_score

# Read the input text from the file
input_file = 'humanized_text.txt'  # Specify the correct file path
input_text = read_text_from_file(input_file)

# Truncate the input text to fit within the model's token limit
truncated_text = truncate_text(input_text)

# Detect if the text is AI-generated or human-written
final_label, human_score, ai_score = detect_ai(truncated_text)

# Display the results in a more meaningful way
print(f"Detection result: {final_label}")
print(f"Human-written Score: {human_score:.2f}")
print(f"AI-generated Score: {ai_score:.2f}")
