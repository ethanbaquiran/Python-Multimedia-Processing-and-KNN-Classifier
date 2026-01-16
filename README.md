# Python-Multimedia-Processing-and-KNN-Classifier
A Python-based multimedia processing system that supports image processing, audio manipulation, and basic image classification. It was developed as part of a university-level programming project and demonstrates object-oriented design, data validation, and algorithmic problem-solving

### Image Processing
- Custom 'RGBImage' class with strict validation and deep-copy support
- Image transofrmations:
  - Negation
  - Grayscale conversion
  - 180 degree rotation
  - Brightness adjustment
- Advanced effects:
  - Pixelation
  - Edge highlighting using convolution-style masking

### Audio Processing
- WAV audio manipulation using raw sample data
- Audio effects:
  - Reverse playback
  - Speed up / slow down
  - Reverb effect
  - Percentage-based audio clipping
 
### Tiered Processing System
- Three service tiers:
  - **Standard**
  - **Premium**
  - **Premium+**
- Cost-tracking system with coupon suport
- Demonstrates inheritance, meethod overriding, and abstraction

### Image classification
- Simple K-Nearest Neighbors (KNN) classifier
- Uses pixel-wise Euclidean distance
- Classifies images based on labeled training data

## Concepts Demonstrated
- Object-Oriented Programming
  - Inheritance
  - Polymorphism
  - Encapsulation
- Defensive programming and error handling
- Image and audio signal processing
- Algorithmic problem solving
- Introductory machine learning (KNN)

## Tech Used
- Python
- NumPy
- Pillow (PIL)
- wave
