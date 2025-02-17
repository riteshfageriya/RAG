# frontend.py
import streamlit as st
from rag_mindflix import VideoQASystem  # Ensure this import is correct
from pytube import YouTube
import tempfile
import time
import logging

# Disable Streamlit's file watcher (if not already done via config.toml)
st.set_option('server.fileWatcherType', 'none')

# Initialize system
@st.cache_resource
def load_system():
    try:
        return VideoQASystem()
    except Exception as e:
        st.error(f"Failed to initialize VideoQASystem: {str(e)}")
        return None

qa_system = load_system()

# Streamlit UI
st.title("🎥 YouTube Video Q&A System")
st.markdown("Analyze any YouTube video content through Q&A")

# Session state for video processing
if 'processed' not in st.session_state:
    st.session_state.processed = False

# URL Input
video_url = st.text_input("Enter YouTube URL:")
process_button = st.button("Process Video")

# Validate YouTube URL
def is_valid_youtube_url(url):
    """Check if the URL is a valid YouTube URL"""
    return "youtube.com" in url or "youtu.be" in url

if process_button and video_url:
    if not is_valid_youtube_url(video_url):
        st.error("Please enter a valid YouTube URL")
    else:
        with st.spinner("Downloading and processing video..."):
            try:
                # Download audio from YouTube
                yt = YouTube(video_url)
                audio_stream = yt.streams.filter(only_audio=True).first()

                with tempfile.TemporaryDirectory() as tmpdir:
                    audio_path = audio_stream.download(output_path=tmpdir)

                    # Transcribe audio
                    transcription = qa_system.process_video(audio_path)

                    if transcription:
                        st.session_state.processed = True
                        st.success(f"Video processed: {yt.title}")
                        st.video(video_url)
                    else:
                        st.error("Failed to process video")

            except Exception as e:
                st.error(f"Error processing video: {str(e)}")

# Reset button
if st.session_state.get('processed'):
    if st.button("Reset and Process New Video"):
        st.session_state.processed = False
        st.experimental_rerun()

# Q&A Interface
if st.session_state.get('processed'):
    question = st.text_input("Ask about the video content:", placeholder="Enter your question here")

    if st.button("Get Answer") and question:
        with st.spinner("Analyzing content..."):
            try:
                start_time = time.time()
                answer = qa_system.ask(question)
                response_time = time.time() - start_time

                st.subheader("Answer:")
                st.markdown(answer)
                st.caption(f"Response time: {response_time:.2f} seconds")

            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")