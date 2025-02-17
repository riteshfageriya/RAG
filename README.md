# Rag MindFlix

Rag MindFlix is a web application that allows users to interact with YouTube videos by asking questions related to the video content. Using the YouTube video link, the app processes the video and provides answers to questions, making it easier for users to gain insights from the video without watching the entire content.

## Features

- **YouTube Video Link Input**: Allows users to input a YouTube video link.
- **Ask Questions**: Users can ask questions related to the video, and the application will provide answers based on the video content.
- **Host on Streamlit**: The application is hosted on Streamlit, providing an interactive and responsive web interface.

## Technologies Used

- **Python**: The core language used for implementing the backend of the application.
- **Streamlit**: A Python framework for building interactive web applications.
- **YouTube API**: For extracting the video content and metadata.
- **OpenAI API**: (or any other NLP model you may be using) for answering user queries based on the video content.
- **Various Python Libraries**: Including `requests`, `pytube`, `beautifulsoup4`, and others for extracting, processing, and interacting with video content.

## How to Use

1. Visit the hosted website at:  
   [Rag MindFlix](https://rag-mindflix-sachin-raj.streamlit.app/)

2. **Input a YouTube Video Link**:  
   Enter a YouTube video link in the provided input field.

3. **Ask Questions**:  
   After the video is processed, you can ask questions related to the video. The app will provide answers based on the video content.

## Installation

To run the application locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository-name.git
   cd your-repository-name
2.Create a virtual environment (optional but recommended):
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  ```
3.Install the required libraries:
  ```bash
   pip install -r requirements.txt
   ```
4.Run the Streamlit app:
   ```bash
   streamlit run frontend.py
   ```
Usage Notes
Make sure to input valid YouTube video links to get accurate results.
The system may take a few seconds to process the video and provide answers to your questions, depending on the video length.


License
This project is licensed under the MIT License - see the LICENSE file for details.


Feel free to explore and contribute to the project!
   ### Customization:
- Replace `[https://github.com/your-username/your-repository-name](https://github.com/schnrj/RAG_Mindflix/tree/main).git` with the actual GitHub repository URL.
- If you have specific installation or usage steps (e.g., API keys), add them as needed in the instructions.

Let me know if you need any adjustments or additions to the README!
