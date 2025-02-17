



import os


os.environ["GEMINI_API_KEY"] = "AIzaSyCcrBaNxQ2hR89WITgIsxHoJZsck8NG650"

print("Updated API Key:", os.environ.get("GEMINI_API_KEY"))

# pip install python-dotenv






import os
from dotenv import load_dotenv


load_dotenv(".env")


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=ftDsSB3F5kg"


import os
import tempfile
from pytube import YouTube
import whisper
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

def extract_video_id(url):
    """Smart URL parser for all YouTube formats"""
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    return parse_qs(urlparse(url).query).get("v", [None])[0]

def get_english_transcript(video_id):
    """Enhanced transcript fetcher with translation"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Priority 1: Existing English subs
        try:
            transcript = transcript_list.find_transcript(['en'])
            return clean_transcript(transcript.fetch()), "existing_en"
        except:
            pass

        # Priority 2: Translated subs
        for transcript in transcript_list:
            if transcript.is_translatable:
                return clean_transcript(transcript.translate('en').fetch()), "translated"

        # Priority 3: Any available subs
        for transcript in transcript_list:
            return clean_transcript(transcript.fetch()), "raw_subtitle"

        raise Exception("No subtitles found")

    except Exception as e:
        print(f"Subtitle error: {str(e)}")
        return None, "error"

def clean_transcript(entries):
    """Improve subtitle formatting"""
    return " ".join([entry['text'].replace('\n', ' ') for entry in entries])

def transcribe_audio(youtube_url):
    """Robust audio transcription with Whisper"""
    try:
        yt = YouTube(youtube_url)
        audio = yt.streams.filter(only_audio=True).first()

        model = whisper.load_model("small")  # Better accuracy than 'base'

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = audio.download(output_path=tmpdir)
            result = model.transcribe(audio_path)
            return result["text"], "audio_transcription"

    except Exception as e:
        return f"Audio failed: {str(e)}", "error"

# Main Execution
youtube_url = "https://www.youtube.com/watch?v=ftDsSB3F5kg"
video_id = extract_video_id(youtube_url)

# Try subtitle methods first
transcript, source = get_english_transcript(video_id)

# Fallback to audio if needed
if not transcript:
    transcript, source = transcribe_audio(youtube_url)

# Save and show results
if transcript and not isinstance(transcript, str):
    with open("transcription.txt", "w") as f:
        f.write(transcript)
    # print(f"Success! Source: {source}")
    # print("\nSample transcript:")
    # print(transcript[:500] + "...")
else:
    print("Failed:")
    


import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import re

# Configure Gemini
genai.configure(api_key="AIzaSyCcrBaNxQ2hR89WITgIsxHoJZsck8NG650")
model = genai.GenerativeModel('gemini-pro')

# 1. Preprocess the transcript
def preprocess_transcript(text):
    # Add newlines after sentence endings
    text = re.sub(r'\. ([A-Z])', r'.\n\1', text)
    # Add paragraph breaks for common video terms
    text = re.sub(r'(\[Music\]|\. )', r'\n\1', text)
    return text

# 2. Load and preprocess transcription
with open("transcription.txt", "r") as f:
    processed_transcript = preprocess_transcript(f.read())

# 3. Configure text splitting for dialogue-heavy content
tokenizer = tiktoken.get_encoding("cl100k_base")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Smaller chunks for dense content
    chunk_overlap=100,
    length_function=lambda text: len(tokenizer.encode(text)),
    separators=["\n", ". ", "! ", "? ", ", ", " then ", " apart from this", " "]
)

# 4. Split transcript
chunks = text_splitter.split_text(processed_transcript)
print(f"Split into {len(chunks)} meaningful chunks")

# 5. Enhanced processing function
def analyze_chunk(chunk):
    prompt = f"""Analyze this video segment focusing on technical film-making aspects:

    {chunk}

    Identify and explain 2-3 key concepts from this segment related to:
    - Directing techniques
    - Team coordination
    - Technical terminology
    - Production workflow

    Present as bullet points:"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Analysis error: {str(e)[:150]}"

try:
    with open("analysis.txt", "w") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"\n{'='*40}\nSegment {i+1} Content:\n{chunk}\n\nAnalysis:")
            analysis = analyze_chunk(chunk)
            f.write(f"\n{analysis}\n")
except Exception as e:
    with open("error_log.txt", "w") as err_file:
        err_file.write(f"Error encountered: {str(e)}\n")



# core_system.py
import os
import re
import time
import logging
import numpy as np
import hashlib
import whisper
import pickle
from pathlib import Path
from collections import defaultdict
from tenacity import retry, wait_exponential, stop_after_attempt
from dotenv import load_dotenv
import cohere
import google.generativeai as genai
from tiktoken import get_encoding
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Manager (updated for your .env)
class Config:
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.embed_model = "embed-english-v3.0"
        self.top_k = 3
        self.cache_dir = Path(".embedding_cache")
        self.rate_limit = 10
        self.max_retries = 5
        self.min_chunk_length = 50
        self.embed_dim = 1024  # Cohere v3 dimension

        load_dotenv()
        self.cohere_key = os.getenv("YOUR_COHERE_API_KEY")
        self.google_key = os.getenv("YOUR_GOOGLE_API_KEY")

# Transcript Processor with your filename
class TranscriptProcessor:
    def __init__(self, config):
        self.config = config
        self.enc = get_encoding("cl100k_base")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=self._token_count,
            separators=["\n", ". ", "! ", "? ", "; ", ", "]
        )

    def _token_count(self, text):
        return len(self.enc.encode(text))

    def preprocess(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        return re.sub(r'(\[Music\]|\(.*?\))', '', text)

    def chunk(self, text):
        preprocessed = self.preprocess(text)
        chunks = self.splitter.split_text(preprocessed)
        return [c for c in chunks if len(c) > self.config.min_chunk_length]

# Embedding Service with dimension fixes
class EmbeddingService:
    def __init__(self, config):
        self.config = config
        self.co = cohere.Client(config.cohere_key)
        config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, text):
        return self.config.cache_dir / f"{hashlib.md5(text.encode()).hexdigest()}.pkl"

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10),
          stop=stop_after_attempt(5))
    def embed(self, texts):
        cached = []
        to_process = []

        # Load cached embeddings as 2D arrays
        for text in texts:
            cache_file = self._cache_path(text)
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        emb = pickle.load(f)
                        if emb.ndim == 1:
                            emb = emb.reshape(1, -1)
                        cached.append(emb)
                except Exception as e:
                    logger.warning(f"Cache load error: {e}")
                    to_process.append(text)
            else:
                to_process.append(text)

        # Process new texts
        new_embeddings = np.zeros((0, self.config.embed_dim))
        if to_process:
            try:
                response = self.co.embed(
                    texts=to_process,
                    model=self.config.embed_model,
                    input_type="search_document"
                )
                new_embeddings = np.array(response.embeddings)

                # Save as 2D arrays
                for text, emb in zip(to_process, new_embeddings):
                    with open(self._cache_path(text), "wb") as f:
                        pickle.dump(emb.reshape(1, -1), f)
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                raise

        # Combine embeddings
        if cached:
            cached_embs = np.vstack(cached)
            if new_embeddings.size > 0:
                return np.vstack([cached_embs, new_embeddings])
            return cached_embs
        return new_embeddings

# Vector Store with dimension handling
class VectorStore:
    def __init__(self, embed_dim):
        self.embeddings = np.zeros((0, embed_dim))
        self.chunks = []
        self.index = defaultdict(list)
        self.transcriber=whisper.load_model("base")

    def process_video(self, audio_path):
        """Transcribe audio from video file"""
        try:
            result = self.transcriber.transcribe(audio_path)
            self.search_engine.index_transcript(result["text"])
            return result["text"]
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None



    def add_documents(self, chunks, embeddings):
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Chunks/embeddings count mismatch")

        self.embeddings = np.vstack([self.embeddings, embeddings])
        self.chunks.extend(chunks)

        # Build search index
        for idx, chunk in enumerate(chunks):
            for word in set(chunk.lower().split()):
                self.index[word].append(idx)

    def search(self, query_embed, top_k=5):
        if self.embeddings.size == 0:
            return []

        query_embed = np.array(query_embed)
        if query_embed.ndim == 1:
            query_embed = query_embed.reshape(1, -1)

        similarities = np.dot(self.embeddings, query_embed.T).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.chunks[i], similarities[i]) for i in top_indices]

# Semantic Search Engine
class SemanticSearch:
    def __init__(self, config):
        self.config = config
        self.processor = TranscriptProcessor(config)
        self.embedder = EmbeddingService(config)
        self.vector_store = VectorStore(config.embed_dim)

    def index_transcript(self, transcript):
        chunks = self.processor.chunk(transcript)
        if not chunks:
            raise ValueError("No valid chunks created")

        embeddings = self.embedder.embed(chunks)
        self.vector_store.add_documents(chunks, embeddings)

    def search(self, query):
        try:
            response = self.embedder.co.embed(
                texts=[query],
                model=self.config.embed_model,
                input_type="search_query"
            )
            query_embed = np.array(response.embeddings[0])
            return self.vector_store.search(query_embed, self.config.top_k)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

# Answer Generator
class AnswerGenerator:
    def __init__(self, config):
        self.config = config
        genai.configure(api_key=config.google_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.last_call = 0

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10),
          stop=stop_after_attempt(3))
    def generate(self, context_chunks, question):
        current_time = time.time()
        if current_time - self.last_call < 60/self.config.rate_limit:
            time.sleep(60/self.config.rate_limit - (current_time - self.last_call))

        context = "\n\n".join([f"Context {i+1}: {chunk}"
                             for i, (chunk, _) in enumerate(context_chunks)])
        prompt = f"""Answer using this context:
        {context}

        Question: {question}

        Guidelines:
        - Be specific about video production terms
        - Use bullet points for key items
        - Mention technical roles

        Answer:"""

        try:
            response = self.model.generate_content(prompt)
            self.last_call = time.time()
            return response.text
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "Could not generate answer"

# Main System
class VideoQASystem:
    def __init__(self):
        self.config = Config()
        self.search_engine = SemanticSearch(self.config)
        self.answer_gen = AnswerGenerator(self.config)

    def process_transcription(self, file_path="transcription.txt"):
        try:
            with open(file_path, "r") as f:
                transcript = f.read()
            self.search_engine.index_transcript(transcript)
            logger.info("Indexed transcription successfully")
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

    def ask(self, question):
        try:
            results = self.search_engine.search(question)
            if not results:
                return "No relevant information found"
            return self.answer_gen.generate(results, question)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return "Error processing question"
    
    def process_video(self, audio_path):
        """Process video audio and index transcription"""
        
        try:
            transcription=self.search_engine.vector_store.process_video(audio_path)
            if transcription:
                self.search_engine.index_transcript(transcription)
                logger.info("Video Processes successfully")
                return transcription
            
            else:
                logger.error("Falied to process the video")
                return None
        except Exception as e:
            logger.error(f"video processing failed:{e}")
            return None
    
   
# Main Execution
if __name__ == "__main__":
    system = VideoQASystem()

    try:
        system.process_transcription()
        print("System ready! Ask about the video content:")

        while True:
            question = input("\nYour question (q to quit): ").strip()
            if question.lower() in ('q', 'quit', 'exit'):
                break
            print(f"\n{system.ask(question)}")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print("System failed to initialize. Check logs and transcription file.")

