from google.cloud import vision

# Initialize the Google Vision API client
client = vision.ImageAnnotatorClient()

# Load your image into memory
with open('4.png', 'rb') as image_file:
    content = image_file.read()
image = vision.Image(content=content)

# Perform text detection on the image
response = client.text_detection(image=image)
texts = response.text_annotations

# Print detected text
for text in texts:
    print(f'Detected text: {text.description}')

# The first element of text_annotations usually contains the entire detected text
if texts:
    print(f'First detected text: {texts[0].description}')
